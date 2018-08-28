package com.linyk3.facedetection;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class FrameDiffDetector {

    private static int imgCount = 0;
    public static void main(String[] args) {
        // 加载OpenCV类库
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        String xmlPath = FrameDiffDetector.class.getResource("haarcascade_frontalface_alt.xml").getPath().substring(1) ; 
        String srcPath = FrameDiffDetector.class.getResource("/img/src/").getPath().substring(1);
        String dstPath = FrameDiffDetector.class.getResource("/img/dst/").getPath().substring(1);
        // 检测图片
        String imgsrc1 = srcPath+ "001.JPG";
        String imgsrc2 = srcPath+ "002.JPG";
        String imgdst1 = dstPath+ "101.JPG";
        String imgdst2 = dstPath+ "102.JPG";
        // 将图片转化为Mat对象
        Mat mat1 = Imgcodecs.imread(imgsrc1);
        Mat mat2 = Imgcodecs.imread(imgsrc2);
        Mat mat3 = new Mat();
        Mat mat4 = new Mat();
        System.out.println(mat1.cols());
        System.out.println(mat1.rows());
        // 重置Mat对象大小
        Imgproc.resize(mat1, mat3, new Size(600, 500));
        Imgproc.resize(mat2, mat4, new Size(600, 500));
        Mat tmp = new Mat();
        // 两个Mat对象差分求值
        Core.absdiff(mat3, mat4, tmp);
        // 计算矩阵元素之和
        Scalar scalar = Core.sumElems(tmp);
        System.out.println(scalar.toString());
        
        if(scalar.val[0] > 777777) {
        	System.out.println("存在运动物体");
        } else {
        	System.out.println("不存在运动物体");
        }
        // 将矩阵转化为图片
        Imgcodecs.imwrite(imgdst1, tmp);
    }
}