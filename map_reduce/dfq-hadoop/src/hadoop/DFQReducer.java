package hadoop;

import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Random;

import javax.imageio.ImageIO;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

/**
 * Với mỗi ảnh:
 * - Đọc ảnh gốc từ local path
 * - Tạo ảnh mới = ảnh gốc + Gaussian noise (nhẹ, giữ nội dung)
 * - Ghi ra thư mục local: outRoot/images/train/<same_name>
 * - Copy label tương ứng từ labelsTrainDir/<same_base>.txt -> outRoot/labels/train/<same_base>.txt
 *
 * Cuối job: viết data.yaml đầy đủ (path/train/val/names)
 */
public class DFQReducer extends Reducer<Text, Text, Text, Text> {

    private File outImagesTrainDir;
    private File outLabelsTrainDir;
    private File outRootDir;
    private File outDataYaml;

    private File labelsTrainDir;
    private double noiseStd;
    private boolean overwrite;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
        String outRoot = ctx.getConfiguration().get("dfq.outRoot");               // ví dụ: D:/Code/BigData/exp_yolo_generated_gd1
        String labelsTrain = ctx.getConfiguration().get("dfq.labelsTrainDir");    // ví dụ: D:/Code/BigData/dataset_yolo/labels/train
        String imagesSplit = ctx.getConfiguration().get("dfq.imagesSplit", "train");
        System.out.println(">>> DFQ RUNNING ON SPLIT = " + imagesSplit);
        this.noiseStd = ctx.getConfiguration().getFloat("dfq.noiseStd", 8.0f);    // độ mạnh noise [mặc định 8]
        this.overwrite = ctx.getConfiguration().getBoolean("dfq.overwrite", true);

        if (outRoot == null || labelsTrain == null) {
            throw new IOException("Missing config: dfq.outRoot or dfq.labelsTrainDir");
        }
        this.outRootDir = new File(outRoot);
        this.outImagesTrainDir = new File(outRootDir, "images/" + imagesSplit);
        this.outLabelsTrainDir = new File(outRootDir, "labels/" + imagesSplit);
        this.outDataYaml = new File(outRootDir, "data.yaml");

        this.labelsTrainDir = new File(labelsTrain);

        // Tạo thư mục output
        if (!outImagesTrainDir.mkdirs() && !outImagesTrainDir.exists())
            throw new IOException("Cannot create " + outImagesTrainDir.getAbsolutePath());
        if (!outLabelsTrainDir.mkdirs() && !outLabelsTrainDir.exists())
            throw new IOException("Cannot create " + outLabelsTrainDir.getAbsolutePath());
    }
    
    private static void sanitizeYoloLabel(File src, File dst, int numClasses) throws IOException {
        if (!src.exists()) { 
            if (!dst.exists()) dst.createNewFile();
            return;
        }
        try (
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(src), "UTF-8"));
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(dst), "UTF-8"))
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] tok = line.split("\\s+");
                if (tok.length != 5) continue;
                try {
                    int cls = Integer.parseInt(tok[0]);
                    double x = Double.parseDouble(tok[1]);
                    double y = Double.parseDouble(tok[2]);
                    double w = Double.parseDouble(tok[3]);
                    double h = Double.parseDouble(tok[4]);
                    boolean fieldsOk = cls >= 0 && cls < numClasses
                            && x >= 0 && x <= 1
                            && y >= 0 && y <= 1
                            && w > 0 && w <= 1
                            && h > 0 && h <= 1;

            boolean inside = (x - w / 2.0) >= 0.0 &&
                             (y - h / 2.0) >= 0.0 &&
                             (x + w / 2.0) <= 1.0 &&
                             (y + h / 2.0) <= 1.0;

            boolean ok = fieldsOk && inside;
                    if (ok) {
                        bw.write(String.format("%d %.6f %.6f %.6f %.6f", cls, x, y, w, h));
                        bw.newLine();
                    }
                } catch (Exception ignore) {}
            }
        }
        if (dst.length() == 0) {
            // Ghi 1 dòng COCO hợp lệ tối thiểu — Ví dụ 1 object giả cực nhỏ
            try (BufferedWriter bw2 = new BufferedWriter(new FileWriter(dst))) {
                bw2.write("0 0.5 0.5 0.01 0.01"); // 1 object "at least exist"
                bw2.newLine();
            }
        }
    }

    @Override
    protected void reduce(Text fileName, Iterable<Text> imagePaths, Context ctx)
            throws IOException, InterruptedException {

        // Ta dùng record đầu tiên (mỗi key 1 record)
        String path = imagePaths.iterator().next().toString();

        File src = new File(path);
        if (!src.exists()) {
            ctx.write(new Text("WARN_MISSING_IMG"), new Text(path));
            return;
        }
        
        File outImg = new File(outImagesTrainDir, fileName.toString());

        try {
            javax.imageio.ImageIO.setUseCache(false); // tránh cache tạm gây lỗi

            BufferedImage img;
            try (InputStream in = new FileInputStream(src)) {
                img = ImageIO.read(in); // có thể fail với CMYK/Gray
            }
            if (img == null) throw new IOException("ImageIO returned null");

            BufferedImage rgbImg = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
            rgbImg.getGraphics().drawImage(img, 0, 0, null);

            if (ctx.getConfiguration().get("dfq.imagesSplit", "train").equals("train")) {
                // TRAIN → thêm noise rồi ghi ra
                BufferedImage out = addGaussianNoise(rgbImg, noiseStd);
                ImageIO.write(out, extOf(fileName.toString()), outImg);
            } else {
                // VAL → GIỮ ẢNH NGUYÊN BẢN 100%, KHÔNG ImageIO, KHÔNG decode/encode
                Files.copy(src.toPath(), outImg.toPath(), StandardCopyOption.REPLACE_EXISTING);
            }

        } catch (Exception ex) {
            Files.copy(src.toPath(), outImg.toPath(), StandardCopyOption.REPLACE_EXISTING);
            ctx.write(new Text("FALLBACK_COPY"), new Text(src.getAbsolutePath()));
        }

        // ✅ TỰ ĐỘNG COPY + LỌC LABEL TƯƠNG ỨNG 
        File srcLabel = new File(labelsTrainDir, baseName(fileName.toString()) + ".txt");
        File dstLabel = new File(outLabelsTrainDir, fileName.toString().replaceAll("\\.[^.]+$", ".txt"));

        sanitizeYoloLabel(srcLabel, dstLabel, 80);
    }

    @Override
    protected void cleanup(Context ctx) throws IOException, InterruptedException {
        // Viết data.yaml đầy đủ (names = COCO80)
        if (!outDataYaml.exists() || overwrite) {
            try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outDataYaml), "UTF-8"))) {
                bw.write("path: " + outRootDir.getAbsolutePath().replace("\\", "/") + "\n");
                bw.write("train: images/train\n");
                bw.write("val: images/val\n"); // nếu bạn muốn tạo val riêng, chạy job lần 2 với imagesSplit=val
                bw.write("names:\n");
                String[] COCO80 = coco80();
                for (int i = 0; i < COCO80.length; i++) {
                    bw.write("  " + i + ": " + COCO80[i] + "\n");
                }
            }
        }
    }

    private static String baseName(String fn) {
        int dot = fn.lastIndexOf('.');
        return dot >= 0 ? fn.substring(0, dot) : fn;
    }

    private static String extOf(String fn) {
        int dot = fn.lastIndexOf('.');
        if (dot >= 0 && dot < fn.length() - 1) return fn.substring(dot + 1);
        return "jpg";
    }

    private static BufferedImage addGaussianNoise(BufferedImage img, double std) {
        int w = img.getWidth();
        int h = img.getHeight();
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Random rnd = new Random();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 255;
                int g = (rgb >> 8)  & 255;
                int b = (rgb)       & 255;
                // noise ~ N(0, std^2)
                r = clamp((int)Math.round(r + rnd.nextGaussian() * std));
                g = clamp((int)Math.round(g + rnd.nextGaussian() * std));
                b = clamp((int)Math.round(b + rnd.nextGaussian() * std));
                int nrgb = (r << 16) | (g << 8) | b;
                out.setRGB(x, y, nrgb);
            }
        }
        return out;
    }

    private static int clamp(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

    // Danh sách 80 lớp COCO
    private static String[] coco80() {
        return new String[]{
            "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
            "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
            "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant",
            "bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
            "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
        };
    }
}
