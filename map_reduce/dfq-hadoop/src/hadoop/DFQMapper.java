package hadoop;

import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/**
 * Input: mỗi dòng là đường dẫn file ảnh (local), ví dụ:
 * D:/Code/BigData/dataset_yolo/images/train/000000000009.jpg
 *
 * Output (key = base filename, val = full path):
 * 000000000009.jpg  D:/.../images/train/000000000009.jpg
 */
public class DFQMapper extends Mapper<LongWritable, Text, Text, Text> {
    @Override
    protected void map(LongWritable key, Text value, Context ctx)
            throws IOException, InterruptedException {
        String imagePath = value.toString().trim();
        if (imagePath.isEmpty()) return;
        String fileName = imagePath.replace("\\", "/");
        fileName = fileName.substring(fileName.lastIndexOf('/') + 1);
        ctx.write(new Text(fileName), new Text(value.toString()));
    }
}
