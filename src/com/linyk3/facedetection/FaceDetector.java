package com.linyk3.facedetection;

import java.text.SimpleDateFormat;
import java.util.Date;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetector {

	private static int imgCount = 0;
    public static void main(String[] args) {
    	//导入opencv,本项目采用的是opencv3.2.0
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("FaceDetector begin");
        //采用opencv320\sources\data\haarcascades 安装目录里的 人脸检测器 haar特征
        //getPath() 打印出来的路径前面多了一个反斜线，导致读不到配置文件
        String xmlPath = FaceDetector.class.getResource("haarcascade_frontalface_alt.xml").getPath().substring(1) ; 
        //System.out.println(xmlPath);
        
        //检测的照片路径
        String srcPath = FaceDetector.class.getResource("/img/src/").getPath().substring(1);
        //检测后保存的路径
        String dstPath = FaceDetector.class.getResource("/img/dst/").getPath().substring(1);
        //System.out.println(srcPath);
        
        String imgsrc = srcPath+"zhuyin.JPG";
        
        System.out.println("detection img: " + imgsrc);
        //核心：加载xml识别配置文件，将图像转化为Mat对象后利用OpenCV函数进行识别，将识别到的人脸保存到 faceDetections
        CascadeClassifier faceDetector = new CascadeClassifier(xmlPath);
        Mat image = Imgcodecs.imread(imgsrc);
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

        SimpleDateFormat formatter  = new SimpleDateFormat("yyyyMMddHHmmss");
        String date = formatter.format(new Date());
        for (Rect rect : faceDetections.toArray()) {
        	//画出人脸方框
            Imgproc.rectangle(image, new Point(rect.x, rect.y),new Point(rect.x + rect.width, rect.y + rect.height),new Scalar(0, 255, 255));
            
            //将识别到的人脸剪切，并重新设置大小
            Rect roi = new Rect(rect.x, rect.y, rect.width, rect.height);
            Mat tmp = new Mat(image, roi);
            Mat dst = tmp.clone();
            Imgproc.resize(tmp, dst, new Size(160, 160));
            
            // 将人脸保存
            String filename = dstPath + date +  imgCount +".png";
            imgCount++;
            System.out.println(String.format("Writing %s", filename));
            Imgcodecs.imwrite(filename, dst);
        }
        String filename = dstPath + date +".png";
        System.out.println(String.format("Writing %s", filename));
        Imgcodecs.imwrite(filename, image);
        
    }
}