package com.linyk3.facedetection;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
/**
 * @author linyk001
 * @date 2018/08/29
 */
public class MovingDetection {

    private static final SimpleDateFormat dateSdf = new SimpleDateFormat("yyyyMMdd");
    private static final SimpleDateFormat datetimeSdf = new SimpleDateFormat("yyyyMMddHHmmss");
    private static String imgPath = "D:\\fras-files\\nfs\\img-mm\\";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat preMat = null;
        Mat curMat = null;
        Mat mat = null;
        for (int i = 1; i < 1295; i++) {
            System.out.println(i);
            String url = "D:\\fras-files\\nfs\\frame\\linyk001\\hiv00006_" + i + ".png";
            curMat = Imgcodecs.imread(url);
            mat = curMat.clone();
            if (preMat == null) {
                preMat = curMat;
                continue;
            }
            detection(preMat, curMat);

            preMat = mat;
        }

    }

    public static void detection(Mat preMat, Mat curMat) {
//        System.out.println(preMat.cols());
//        System.out.println(curMat.cols());
        Mat tmpMat = new Mat();
        Mat grayMat = new Mat();
        Core.absdiff(preMat, curMat, tmpMat);
        Imgproc.cvtColor(tmpMat, grayMat, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(grayMat, grayMat, new Size(21, 21), 0);
        Scalar scalar = Core.sumElems(tmpMat);
//        Imgcodecs.imwrite("D:\\cmft\\workspace\\face-detection\\bin\\img\\src\\hiv00006_35_35.png", tmpMat);
        toImage(grayMat, "img1-2", 1);
        Mat thresh = new Mat();
        // 图像二值化
        Imgproc.threshold(grayMat, thresh, 25, 255, Imgproc.THRESH_BINARY);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        // 查找图像轮廓
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        // 膨胀
        Imgproc.dilate(thresh, thresh, new Mat(), new Point(-1, -1), 2);
        int cc = 1;
        for (MatOfPoint mf : contours) {
            // 图像轮廓面积
            if (Imgproc.contourArea(mf) < 2000) {
                continue;
            }
            // 绘制轮廓
            // Imgproc.drawContours(curMat, contours, contours.indexOf(mf), new Scalar(0, 255, 255));
            // 填充颜色
            // Imgproc.fillConvexPoly(curMat, mf, new Scalar(0, 255, 255));
            // 矩形边框
            Rect r = Imgproc.boundingRect(mf);
            // 裁剪矩形
            Mat boundingRect = new Mat(curMat, r);
            String url = "D:\\cmft\\workspace\\face-detection\\bin\\img\\src\\hiv00006_" + cc + ".png";
//            Imgcodecs.imwrite(url, boundingRect);

            // 识别裁剪后的矩形中的人脸
            MatOfRect faceDetections = new MatOfRect();
            CascadeClassifier faceDetector = new CascadeClassifier(
                "D:\\cmft\\workspace\\face-detection\\src\\com\\linyk3\\facedetection\\haarcascade_frontalface_alt.xml");
            faceDetector.detectMultiScale(boundingRect, faceDetections);
            int count = 0;
            for (Rect rect : faceDetections.toArray()) {
                int x = Math.max(1, rect.x - 20);
                int y = Math.max(1, rect.y - 20);
                int w = Math.min(boundingRect.cols() - 1 - x, rect.width + 20 * 2);
                int h = Math.min(boundingRect.rows() - 1 - y, rect.height + 20 * 2);
                Rect roi = new Rect(x, y, w, h);
                Mat tmp = new Mat(boundingRect, roi);
                Imgproc.resize(tmp, tmp, new Size(160, 160));
                toImage(tmp, "img1", count++);
            }

            Imgproc.rectangle(curMat, r.tl(), r.br(), new Scalar(0, 255, 0), 2);
            cc++;

        }
        toImage(curMat, "img2", 1);
        tmpMat.release();
        preMat.release();
    }

    public static void toImage(Mat dst, String url, int count) {
        String day = dateSdf.format(new Date());
        StringBuilder img_url = new StringBuilder();
        img_url.append(imgPath).append(day).append("/").append(url);
        File file = new File(img_url.toString());
        if (!file.exists()) {
            file.mkdirs();
        }
        img_url.append("/").append(datetimeSdf.format(new Date())).append("_").append(Integer.toString(count))
            .append(".png");
        System.out.println(img_url);
        Imgcodecs.imwrite(img_url.toString(), dst);
    }
}
