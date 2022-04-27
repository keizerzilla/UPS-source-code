/**
 * arg[1] = target cloud
 * arg[2] = source cloud
 * arg[3] = theta_x
 * arg[4] = theta_y
 * arg[5] = theta_z
 * arg[6] = size_min
 * arg[7] = size_max
 * arg[8] = icp_it
 * arg[9] = icp_stop
 * arg[10] = output_txt
 */

// For tests, please change the fixed axes (source and target) to partitionate the cloud

/* Includes */
#define PI 3.14159265
#include <algorithm>
#include <string>		// Necessário para usar strings
#include <vector>
#include <math.h>
#include <iostream>
#include <ctime>
#include <fstream>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/registration.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/common/common.h>
#include <pcl/console/time.h> 
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/console/print.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/pca.h>
#include <bits/stdc++.h>



/* Defines */
//#define PI 3.14159265

/* Namespace and typedefs */
using namespace std;
using std::vector;

typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

typedef enum{X, Y, Z} axis;

/* Function signatures */
double computeRMSE(pcl::PointCloud<pcl::PointXYZ>::ConstPtr target,
                   pcl::PointCloud<pcl::PointXYZ>::ConstPtr source,
                   double max_range);

bool compareX(pcl::PointXYZ p1, pcl::PointXYZ p2);

bool compareY(pcl::PointXYZ p1, pcl::PointXYZ p2);

bool compareZ(pcl::PointXYZ p1, pcl::PointXYZ p2);

//bool compareTarget(pcl::PointXYZ p1, pcl::PointXYZ p2);

//bool compareSource(pcl::PointXYZ p1, pcl::PointXYZ p2);

pcl::PointXYZ variance(PointCloud cloud, Eigen::Vector4f centroid);

vector<PointCloud::Ptr> cloudPartitionate(PointCloud::Ptr cloud, int num_part);

/* Main function */
int main (int argc, char** argv)
{
    clock_t begin;
    begin = clock();

    PointCloud::Ptr source (new PointCloud);
    PointCloud::Ptr target (new PointCloud);
    PointCloud::Ptr target_u (new PointCloud);

    // Load target cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *target) == -1) {
        PCL_ERROR ("Couldn't read target cloud\n");
        return (-1);
    }

    // Load source cloud
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[2], *source) == -1) {
        PCL_ERROR ("Couldn't read source cloud\n");
        return (-1);
    }

    // Load theta
    float theta_x = atof(argv[3]);
    float theta_y = atof(argv[4]);
    float theta_z = atof(argv[5]);

    // Load partitions size
    //int maxCloudSize = source->width > target->width ? source->width : target->width;
    int maxCloudSize = target->width;
    int kmin = round(maxCloudSize / atof(argv[7]));
    int kmax = round(maxCloudSize / atof(argv[6]));

    cout << "k_min:  " << kmin << endl;
    cout << "k_max:  " << kmax << endl;

    // Find misaligned target
    Eigen::Affine3f trans_x = Eigen::Affine3f::Identity();
    Eigen::Affine3f trans_y = Eigen::Affine3f::Identity();
    Eigen::Affine3f trans_z = Eigen::Affine3f::Identity();


    trans_x.rotate (Eigen::AngleAxisf (theta_x, Eigen::Vector3f::UnitZ()));
    trans_y.rotate (Eigen::AngleAxisf (theta_y, Eigen::Vector3f::UnitY()));
    trans_z.rotate (Eigen::AngleAxisf (theta_z, Eigen::Vector3f::UnitX()));

    pcl::transformPointCloud (*target, *target_u, trans_x);
    pcl::transformPointCloud (*target_u, *target_u, trans_y);
    pcl::transformPointCloud (*target_u, *target_u, trans_z);

    // Calculate reference RMSE
    double rmse_ref = computeRMSE(target,
                                  target_u,
                                  numeric_limits<double>::max ());

    double mse_ref = rmse_ref * rmse_ref;
    // Uncomment if you want to print the rmse_ref value
    cout << "RMSE_ref:  " << rmse_ref << endl;


    // Find best order to test axes
    Eigen::Vector4f centroid_s;
    pcl::compute3DCentroid(*source, centroid_s);
    pcl::PointXYZ var_s = variance(*target, centroid_s);
    // cout << "Source variance " << var_s << endl;
    vector<int> axes_s (3);

    if (var_s.x >= var_s.y && var_s.x >= var_s.z) {
        axes_s[0] = X;
        if (var_s.y >= var_s.z) {
            axes_s[1] = Y;
            axes_s[2] = Z;
        } else {
            axes_s[1] = Z;
            axes_s[2] = Y;
        }
    } else if (var_s.y >= var_s.z) {
        axes_s[0] = Y;
        if (var_s.x >= var_s.z) {
            axes_s[1] = X;
            axes_s[2] = Z;
        } else {
            axes_s[1] = Z;
            axes_s[2] = X;
        }
    } else {
        axes_s[0] = Z;
        if (var_s.x >= var_s.y) {
            axes_s[1] = X;
            axes_s[2] = Y;
        } else {
            axes_s[1] = Y;
            axes_s[2] = X;
        }
    }
    // cout << "Source axes " << axes_s << endl;
    Eigen::Vector4f centroid_t;
    pcl::compute3DCentroid(*target, centroid_t);
    pcl::PointXYZ var_t = variance(*target, centroid_t);

    //  cout << "Target variance " << var_t << endl;

    vector<int> axes_t (3);

    if (var_t.x >= var_t.y && var_t.x >= var_t.z) {
        axes_t[0] = X;
        if (var_t.y >= var_t.z) {
            axes_t[1] = Y;
            axes_t[2] = Z;
        } else {
            axes_t[1] = Z;
            axes_t[2] = Y;
        }
    } else if (var_t.y >= var_t.z) {
        axes_t[0] = Y;
        if (var_t.x >= var_t.z) {
            axes_t[1] = X;
            axes_t[2] = Z;
        } else {
            axes_t[1] = Z;
            axes_t[2] = X;
        }
    } else {
        axes_t[0] = Z;
        if (var_t.x >= var_t.y) {
            axes_t[1] = X;
            axes_t[2] = Y;
        } else {
            axes_t[1] = Y;
            axes_t[2] = X;
        }
    }
    // cout << "Target axes " << axes_t << endl;
    double rmse_min = -1.0;
    Eigen::Matrix4f rt_min;

    PointCloud::Ptr aux_cloud (new PointCloud);

    bool stop = false;

    int final_axis_t;
    int final_axis_s;
    // int final_axis;

    /*for (int i = 0; i < 3; i++) {
                cout << axes_t[i] << ", " << axes_s[i] << endl;
        }*/

    // Loop for each partition size

    for(int k=kmax; k<kmin+1; k++){


        //  for(int k=kmax-1; k>0;k=k-1){
        //           for(int k= kmin; k< kmax;k=k-1){
        //     for (int k = kmax; k >= kmin; k--) {
        // Sort and partitionate in X
        sort(source->points.begin(), source->points.end(), compareX);
        vector<PointCloud::Ptr> partitionsx_s = cloudPartitionate(source, k);

        sort(target->points.begin(), target->points.end(), compareX);
        vector<PointCloud::Ptr> partitionsx_t = cloudPartitionate(target, k);


        // Sort and partitionate in Y
        sort(source->points.begin(), source->points.end(), compareY);
        vector<PointCloud::Ptr> partitionsy_s = cloudPartitionate(source, k);

        sort(target->points.begin(), target->points.end(), compareY);
        vector<PointCloud::Ptr> partitionsy_t = cloudPartitionate(target, k);



        // Sort and partitionate in Z
        sort(source->points.begin(), source->points.end(), compareZ);
        vector<PointCloud::Ptr> partitionsz_s = cloudPartitionate(source, k);

        sort(target->points.begin(), target->points.end(), compareZ);
        vector<PointCloud::Ptr> partitionsz_t = cloudPartitionate(target, k);

        // Loop for each partition
        for(int i=k-1; i>0;i=i-1){
            //    for(int i=k; i>0;i=i-1){

            //  for (int i = 1; i < k; i--) {
            //std::string name_out =  "./luis_" +  std::to_string(i)  + ".pcd";
            // pcl::io::savePCDFileASCII (name_out , *partitionsz_s[i]);
            // pcl::io::savePCDFileASCII ("slice_conv.pcd" , *partitionsz_s[i]);

            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    //
                    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                    // ICP
                    switch (axes_s[p]) {
                    case X:
                        icp.setInputSource(partitionsx_s[i]);
                        break;
                    case Y:
                        icp.setInputSource(partitionsy_s[i]);
                        break;
                    case Z:
                        icp.setInputSource(partitionsz_s[i]);
                        break;

                    }

                    switch (axes_t[q]) {
                    case X:
                        icp.setInputTarget(partitionsx_t[i]);
                        break;
                    case Y:
                        icp.setInputTarget(partitionsy_t[i]);
                        break;
                    case Z:
                        icp.setInputTarget(partitionsz_t[i]);
                        break;

                    }

                    //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);



                    // Set ICP params
                    icp.setMaximumIterations(atoi(argv[8]));
                    //                    if (atoi(argv[9]) > 0) {
                    //                        icp.setEuclideanFitnessEpsilon(mse_ref/atoi(argv[9]));
                    //                    }

                    icp.align(*aux_cloud);

                    // Transform source cloud
                    Eigen::Matrix4f rt = icp.getFinalTransformation();
                    pcl::transformPointCloud(*source, *aux_cloud, rt);

                    // Check RMSE
                    double rmse = computeRMSE(target, aux_cloud, numeric_limits<double>::max ());
                    if (rmse < rmse_min || rmse_min == -1.0) {
                        //final_axis = axes_s[p];
                        final_axis_s = axes_s[p];
                        final_axis_t = axes_t[q];
                        rmse_min = rmse;
                        rt_min = rt;
                    }

                    if (rmse_min <= rmse_ref) {
                        stop = true;
                        break;
                    }

                }
                if (stop) {
                    break;
                }
            }
            if (stop) {
                break;
            }
        }

        if (stop) {
            break;
        }

    }
    // If stop is true the stopping criterion was reached,
    // so aux has the aligned cloud
    if (!stop) {
        pcl::transformPointCloud(*source, *aux_cloud, rt_min);
    }

    double total_time = (clock() - begin) / (double)CLOCKS_PER_SEC;

    Eigen::Matrix3f r (rt_min.block<3,3>(0, 0));
    Eigen::AngleAxisf angle_axis;
    angle_axis.fromRotationMatrix(r);

    ofstream output;
    output.open(argv[9]);
    output << "time: " << total_time << endl;
    output << "min partitions: " << kmin << endl;
    output << "max partitions: " << kmax << endl;
    //output << "final source" << final_axis <<endl;

    output << "final source axis: " << final_axis_s <<endl;
    output << "final target axis: " << final_axis_t <<endl;
    output << "rmse: " << rmse_min << endl;
    output << "rmse reference: " << rmse_ref << endl;
    output << "rt: \n" << rt_min << endl;
    output << "angle: " << (angle_axis.angle()*180.0)/M_PI << endl;
    output << "axis: \n" << angle_axis.axis() << endl;
    output.close();

    double elem1, elem2, elem3, angle123,axis_angle;
    elem1 = rt_min(0,0);

    elem2 = rt_min(1,1);
    elem3= rt_min(2,2);
    angle123= (elem1+elem2+elem3-1)/2;
    axis_angle = acos (angle123) * 180.0 / PI;

    cout << "Rotacao " << axis_angle<<":  " << endl;
    // Visualization
    int sizePoints = 2;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("INPUTS"));

    viewer->setBackgroundColor(255, 255, 255);

    viewer->addPointCloud(target, "target");
    viewer->addPointCloud(source, "source");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,0,0,255,"source");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, sizePoints, "source");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,0, 255, 0,"target");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, sizePoints, "target");


    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_align (new pcl::visualization::PCLVisualizer ("ALIGNED"));

    viewer_align->setBackgroundColor(255,255,255);

    viewer_align->addPointCloud(target, "target");
    viewer_align->addPointCloud(aux_cloud, "aligned");

    viewer_align->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,0,0,255,"aligned");
    viewer_align->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, sizePoints, "aligned");
    viewer_align->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,0, 255, 0,"target");
    viewer_align->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, sizePoints, "target");

    while(!viewer_align->wasStopped()) {
        viewer_align->spinOnce();
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}

// Functions implementation

bool compareX(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
    return (p1.x < p2.x) ? true : false;
}

bool compareY(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
    return (p1.y < p2.y) ? true : false;
}

bool compareZ(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
    return (p1.z < p2.z) ? true : false;
}

/*bool compareTarget(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
        Eigen::Vector3f p1e = p1.getVector3fMap();
        Eigen::Vector3f p2e = p2.getVector3fMap();
        
        float dot_prod_p1 = axis_target.dot(p1e);
        float dot_prod_p2 = axis_target.dot(p2e);
        return (dot_prod_p1 < dot_prod_p2) ? true : false;
}

bool compareSource(pcl::PointXYZ p1, pcl::PointXYZ p2)
{
        Eigen::Vector3f p1e = p1.getVector3fMap();
        Eigen::Vector3f p2e = p2.getVector3fMap();
        
        float dot_prod_p1 = axis_source.dot(p1e);
        float dot_prod_p2 = axis_source.dot(p2e);
        return (dot_prod_p1 < dot_prod_p2) ? true : false;
}*/

double computeRMSE(pcl::PointCloud<pcl::PointXYZ>::ConstPtr target,
                   pcl::PointCloud<pcl::PointXYZ>::ConstPtr source,
                   double max_range)
{
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    tree->setInputCloud(target);

    double fitness_score = 0.0;

    vector<int> nn_indices (1);
    vector<float> nn_dists (1);

    // For each point in the source dataset
    int nr = 0;
    for (size_t i = 0; i < source->points.size (); ++i){
        //Avoid NaN points as they crash nn searches
        if(!pcl_isfinite((*source)[i].x)){
            continue;
        }

        // Find its nearest neighbor in the target
        tree->nearestKSearch (source->points[i], 1, nn_indices, nn_dists);

        // Deal with occlusions (incomplete targets)
        if (nn_dists[0] <= max_range*max_range){
            // Add to the fitness score
            fitness_score += nn_dists[0];
            nr++;
        }
    }

    if (nr > 0)
        return sqrt(fitness_score / nr);

    return (numeric_limits<double>::max ());
}

pcl::PointXYZ variance(PointCloud cloud, Eigen::Vector4f centroid)
{
    pcl::PointXYZ variance;

    variance.x = 0.0;
    variance.y = 0.0;
    variance.z = 0.0;
    for (int i = 0; i < cloud.width; i++) {
        variance.x += (cloud.points[i].x - centroid[0]) * (cloud.points[i].x - centroid[0]);
        variance.y += (cloud.points[i].y - centroid[1]) * (cloud.points[i].y - centroid[1]);
        variance.z += (cloud.points[i].z - centroid[2]) * (cloud.points[i].z - centroid[2]);
    }

    variance.x /= cloud.width - 1;
    variance.y /= cloud.width - 1;
    variance.z /= cloud.width - 1;

    return variance;
}

vector<PointCloud::Ptr> cloudPartitionate(PointCloud::Ptr cloud, int num_part)
{
    // Number of points per partition
    int part_size = floor(cloud->width/num_part);

    vector<PointCloud::Ptr> partitions;
    for (int i = 0; i < num_part; i++) {
        PointCloud::Ptr tmp (new PointCloud);
        tmp->width  = part_size;
        tmp->height = 1;

        for (int j = i*part_size; j < (i+1) * part_size; j++) {
            tmp->points.push_back(cloud->points[j]);
        }

        partitions.push_back(tmp);
    }

    return partitions;
}

