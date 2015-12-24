package crt.mori.second_eyes;

import android.graphics.Path;
import android.util.Log;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvException;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;

import static org.opencv.calib3d.Calib3d.findHomography;
import static org.opencv.video.Video.estimateRigidTransform;

/**
 * Various methods of extracting path from images, matrices, etc. It is a simplified wrapper for the
 * OpenCV library as each function only returns a path (which can be drawn) of the differences
 * between two images/matrices, and takes in various modifiers it need to calculate that difference
 * with different methods.
 */
public class ExtractPath {

    private static final String TAG="ExtractPath";

    private FeatureDetector mFeatureDectector;
    private DescriptorExtractor mDescExtractor;
    private DescriptorMatcher mDescMatcher;
    private MatOfKeyPoint mKeyPointsPrev;
    private MatOfPoint2f prevPoints2f;
    private Scalar mKeyPointsColor = new Scalar(0.5,0.0,0.0,1.0);

    public ExtractPath() {
        super();

        mKeyPointsPrev = new MatOfKeyPoint();
        // set up feature detection
        try {
            mFeatureDectector = FeatureDetector.create(FeatureDetector.FAST);
        } catch (UnsatisfiedLinkError err) {
            Log.e(TAG, "Feature detector failed with");
            err.printStackTrace();
        }
        // set up description detection
        mDescExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
        mDescMatcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
    }

    /**
     * Extract object movement from current and previous frame using Rigid Transformation and save
     * it to path passed to function.
     *
     * @param currFrame Current frame
     * @param prevFrame Previous frame
     * @param path Path variable to which object movement is saved
     * @return Current frame for displaying
     */
    public Path withRigidTransformation(Mat currFrame, Mat prevFrame, Path path) {
        Mat movement = estimateRigidTransform(currFrame, prevFrame, false);
        if (!movement.empty()) {
            float fX = (float) (movement.get(0, 2)[0]);
            float fY = (float) (movement.get(1, 2)[0]);
            Log.i(TAG, "Adding to path (" + Float.toString(fX) + "," + Float.toString(fY) + ")");
            path.rLineTo(fX, fY);
        }
        return path;
    }

    public Path withFASTandHomography(Mat currFrame, Mat mOutFrame, Mat prevDescPoints, Path path) {
        Mat DescPoints = new Mat();
        MatOfDMatch matchingPoints = new MatOfDMatch();
        MatOfKeyPoint currKeyPoints = new MatOfKeyPoint();
        float fPointX, fPointY;

        mFeatureDectector.detect(currFrame, currKeyPoints);
        mDescExtractor.compute(currFrame, currKeyPoints, DescPoints);

        DescPoints.copyTo(prevDescPoints);
        Features2d.drawKeypoints(currFrame, currKeyPoints, mOutFrame, mKeyPointsColor, 0);

        // Convert MatOfKeyPoint to MatOfPoint2f
        KeyPoint[] kpoints = currKeyPoints.toArray();
        Point[] points = new Point[kpoints.length];
        for (int i = 0; i < kpoints.length; ++i) {
            points[i] = new Point(0, 0);
            points[i].x = kpoints[i].pt.x;
            points[i].y = kpoints[i].pt.y;
        }
        MatOfPoint2f currPoints2f = new MatOfPoint2f(points);

        if (prevPoints2f != null) {
            //Features2d.drawKeypoints(mGrayFrame, mKeyPointsPrev, mOutFrame, mKeyPointsColor, 0);

            /*MatOfByte matches = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            calcOpticalFlowPyrLK(mGrayFramePrev,mGrayFrame,prevPoints2f, currPoints2f, matches, err);

            // remove points where LK tracking failed or that went missing
            for(int i=0; i < matches.toArray().length; ++i)
            {
                Point pt = currPoints2f.toArray()[i];
                if ((matches.toArray()[i] == 0)||(pt.x<0)||(pt.y<0))	{
                    currPoints2f.toList().remove(pt);
                }
            }
*/
            // now lets get out rigidTransformation for orientation vector
            try {
                Mat movement = findHomography(currPoints2f, prevPoints2f, Calib3d.LMEDS, 1.2f);
                Log.i(TAG, "We have points. x=" + movement.get(0, 2).toString() + "y=" + movement.get(1, 2).toString());

                fPointX = (float) (movement.get(0, 2)[0]) * 10000000000000.0f;
                fPointY = (float) (movement.get(1, 2)[0]) * 10000000000000.0f;
                Log.i(TAG, "Adding to path (" + Float.toString(fPointX) + "," + Float.toString(fPointY) + ")");
                path.rLineTo(fPointX, fPointY);
            } catch (CvException e) {
                Log.i(TAG, "Vectors are the same");
                //e.printStackTrace();
            }
        }
        currKeyPoints.copyTo(mKeyPointsPrev);
        prevPoints2f = new MatOfPoint2f(points);

        return path;
    }
}
