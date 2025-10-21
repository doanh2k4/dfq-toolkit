package hadoop;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * Usage:
 *  hadoop jar dfq-hadoop.jar hadoop.Driver <HDFS_INPUT> <HDFS_OUTPUT> <LOCAL_OUT_ROOT> <YOLO_ROOT> [imagesSplit=train|val]
 *
 * Ví dụ:
 *  hadoop jar dfq-hadoop.jar hadoop.Driver /dfq/input /dfq/output D:/Code/BigData/exp_yolo_generated_gd1 D:/Code/BigData/dataset_yolo train
 */
public class Driver {
    public static void main(String[] args) throws Exception {
        if (args.length < 4 || args.length > 5) {
            System.err.println("Usage: Driver <HDFS_INPUT> <HDFS_OUTPUT> <LOCAL_OUT_ROOT> <YOLO_ROOT> [imagesSplit=train|val]");
            System.exit(1);
        }

        String hdfsIn  = args[0];
        String hdfsOut = args[1];
        String outRoot = args[2];
        String yoloRoot = args[3];
        String imagesSplit = (args.length == 5) ? args[4] : "train";

        Configuration conf = new Configuration();
        conf.set("dfq.outRoot", outRoot.replace("\\", "/"));
        conf.set("dfq.imagesSplit", imagesSplit);

        // labelsTrainDir = <YOLO_ROOT>/labels/<split>
        conf.set("dfq.labelsTrainDir", (yoloRoot + "/labels/" + imagesSplit).replace("\\", "/"));

        // optional knobs
        conf.setFloat("dfq.noiseStd", 8.0f);     // tăng/giảm độ nhiễu
        conf.setBoolean("dfq.overwrite", true);

        Job job = Job.getInstance(conf, "DFQ YOLO GD1 Generator (" + imagesSplit + ")");
        job.setJarByClass(Driver.class);

        job.setMapperClass(DFQMapper.class);
        job.setReducerClass(DFQReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(hdfsIn));
        FileOutputFormat.setOutputPath(job, new Path(hdfsOut));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
