package com.shekhar.facedetection;

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
    	//����opencv,����Ŀ���õ���opencv3.2.0
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("FaceDetector begin");
        //����opencv320\sources\data\haarcascades ��װĿ¼��� ��������� haar����
        //getPath() ��ӡ������·��ǰ�����һ����б�ߣ����¶����������ļ�
        String xmlPath = FaceDetector.class.getResource("haarcascade_frontalface_alt.xml").getPath().substring(1) ; 
        //System.out.println(xmlPath);
        
        //������Ƭ
        String srcPath = FaceDetector.class.getResource("/img/src/").getPath().substring(1);
        String dstPath = FaceDetector.class.getResource("/img/dst/").getPath().substring(1);
        //System.out.println(srcPath);
        
        //String imgsrc = srcPath+"zhuyin.JPG";
        String imgsrc = srcPath+"many.JPG";
        System.out.println("detection img: " + imgsrc);
        CascadeClassifier faceDetector = new CascadeClassifier(xmlPath);
        Mat image = Imgcodecs.imread(imgsrc);

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image, faceDetections);

        System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

        for (Rect rect : faceDetections.toArray()) {
        	//������������
            //Imgproc.rectangle(image, new Point(rect.x, rect.y),new Point(rect.x + rect.width, rect.y + rect.height),new Scalar(0, 255, 255));

            Rect roi = new Rect(rect.x, rect.y, rect.width, rect.height);
            Mat tmp = new Mat(image, roi);
            Mat dst = tmp.clone();
            Imgproc.resize(tmp, dst, new Size(160, 160));
            SimpleDateFormat formatter  = new SimpleDateFormat("yyyyMMddHHmmss");
            String date = formatter.format(new Date());
            String filename = dstPath + date +  imgCount +".png";
            imgCount++;
            System.out.println(String.format("Writing %s", filename));
            Imgcodecs.imwrite(filename, dst);

        }
    }
}