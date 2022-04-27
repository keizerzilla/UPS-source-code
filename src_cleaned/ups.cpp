/**
 * arg[1]  = target cloud
 * arg[2]  = source cloud
 * arg[3]  = theta_x
 * arg[4]  = theta_y
 * arg[5]  = theta_z
 * arg[6]  = size_min
 * arg[7]  = size_max
 * arg[8]  = icp_it
 * arg[9]  = icp_stop
 * arg[10] = output_txt
 */

#define PI 3.14159265

#include <ctime>
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <bits/stdc++.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/pca.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/console/time.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/console/print.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/registration.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using std::vector;

typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;
typedef PointCloud PointCloud;
typedef enum{X, Y, Z} axis;

bool compareX(pcl::PointXYZ p1, pcl::PointXYZ p2) {
    return p1.x < p2.x;
}

bool compareY(pcl::PointXYZ p1, pcl::PointXYZ p2) {
    return p1.y < p2.y;
}

bool compareZ(pcl::PointXYZ p1, pcl::PointXYZ p2) {
    return p1.z < p2.z;
}

double computeRMSE(PointCloud::ConstPtr target, PointCloud::ConstPtr source, double max_range) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    
    tree->setInputCloud(target);
    
    double fitness_score = 0.0;
    vector<int> nn_indices(1);
    vector<float> nn_dists(1);
    
    int nr = 0;
    for (size_t i = 0; i < source->points.size (); ++i) {
        // Ignora pontos setados como NaN
        if(!pcl_isfinite((*source)[i].x))
            continue;
        
        // Procura o ponto mais próximo em target
        tree->nearestKSearch (source->points[i], 1, nn_indices, nn_dists);
        
        // Lida com oclusões (targets incompletas)
        if (nn_dists[0] <= max_range*max_range) {
            fitness_score += nn_dists[0];
            nr++;
        }
    }
    
    if (nr > 0)
        return sqrt(fitness_score / nr);
    
    return numeric_limits<double>::max();
}

pcl::PointXYZ variance(PointCloud cloud, Eigen::Vector4f centroid) {
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

vector<PointCloud::Ptr> cloudPartitionate(PointCloud::Ptr cloud, int num_part) {
    // Calcula número de pontos por partição
    int part_size = floor(cloud->width / num_part);
    vector<PointCloud::Ptr> partitions;
    
    for (int i = 0; i < num_part; i++) {
        PointCloud::Ptr tmp(new PointCloud);
        tmp->width = part_size;
        tmp->height = 1;
        
        for (int j = i*part_size; j < (i+1) * part_size; j++)
            tmp->points.push_back(cloud->points[j]);
        
        partitions.push_back(tmp);
    }
    
    return partitions;
}

int main (int argc, char** argv) {
    // Inicia cronômetro
    clock_t begin;
    begin = clock();
    
    // Nuvens
    PointCloud::Ptr source(new PointCloud);
    PointCloud::Ptr target(new PointCloud);
    PointCloud::Ptr target_u(new PointCloud);
    
    // Carrega target
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *target) == -1) {
        PCL_ERROR("Couldn't read target cloud\n");
        return -1;
    }
    
    // Carrega source
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *source) == -1) {
        PCL_ERROR("Couldn't read source cloud\n");
        return -1;
    }
    
    // Captura microdesalinhamento: existe um para cada eixo
    float theta_x = atof(argv[3]);
    float theta_y = atof(argv[4]);
    float theta_z = atof(argv[5]);
    
    // Load partitions size
    int maxCloudSize = target->width; // antes havia uma escolha entre src e tgt
    int kmin = round(maxCloudSize / atof(argv[7]));
    int kmax = round(maxCloudSize / atof(argv[6]));
    
    cout << "k_min:  " << kmin << endl;
    cout << "k_max:  " << kmax << endl;
    
    // Microdesalinha nuvem target
    Eigen::Affine3f trans_x = Eigen::Affine3f::Identity();
    Eigen::Affine3f trans_y = Eigen::Affine3f::Identity();
    Eigen::Affine3f trans_z = Eigen::Affine3f::Identity();
    
    trans_x.rotate(Eigen::AngleAxisf(theta_x, Eigen::Vector3f::UnitZ()));
    trans_y.rotate(Eigen::AngleAxisf(theta_y, Eigen::Vector3f::UnitY()));
    trans_z.rotate(Eigen::AngleAxisf(theta_z, Eigen::Vector3f::UnitX()));
    
    pcl::transformPointCloud(*target, *target_u, trans_x);
    pcl::transformPointCloud(*target_u, *target_u, trans_y);
    pcl::transformPointCloud(*target_u, *target_u, trans_z);
    
    // Calcula RMSE de referência
    double rmse_ref = computeRMSE(target, target_u, numeric_limits<double>::max());
    double mse_ref = rmse_ref * rmse_ref;
    
    cout << "rmse_ref:  " << rmse_ref << endl;
    
    // Encontra melhor ordem de teste entre os eixos X, Y, e Z
    Eigen::Vector4f centroid_s;
    pcl::compute3DCentroid(*source, centroid_s);
    pcl::PointXYZ var_s = variance(*target, centroid_s);
    
    cout << "var_s:  " << var_s << endl;
    
    vector<int> axes_s(3);
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
    
    double rmse_min = -1.0;
    Eigen::Matrix4f rt_min;
    PointCloud::Ptr aux_cloud(new PointCloud);
    bool stop = false;
    int final_axis;
    
    // Itera sobre cada partição
    for (int k = kmax; k < kmin+1; k++) {
        // Ordena e particiona em X
        sort(source->points.begin(), source->points.end(), compareX);
        vector<PointCloud::Ptr> partitionsx_s = cloudPartitionate(source, k);
        
        sort(target->points.begin(), target->points.end(), compareX);
        vector<PointCloud::Ptr> partitionsx_t = cloudPartitionate(target, k);
        
        // Ordena e particiona em Y
        sort(source->points.begin(), source->points.end(), compareY);
        vector<PointCloud::Ptr> partitionsy_s = cloudPartitionate(source, k);
        
        sort(target->points.begin(), target->points.end(), compareY);
        vector<PointCloud::Ptr> partitionsy_t = cloudPartitionate(target, k);
        
        // Ordena e particiona em Z
        sort(source->points.begin(), source->points.end(), compareZ);
        vector<PointCloud::Ptr> partitionsz_s = cloudPartitionate(source, k);
        
        sort(target->points.begin(), target->points.end(), compareZ);
        vector<PointCloud::Ptr> partitionsz_t = cloudPartitionate(target, k);
        
        // Itera sobre cada partição
        for (int i = k-1; i > 0; i = i-1) {
            for (int p = 0; p < 3; p++) {
                // Cria objeto ICP e define source e target com base no eixo
                pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                
                switch (axes_s[p]) {
                    case X:
                        icp.setInputSource(partitionsx_s[i]);
                        icp.setInputTarget(partitionsx_t[i]);
                        break;
                    case Y:
                        icp.setInputSource(partitionsy_s[i]);
                        icp.setInputTarget(partitionsy_t[i]);
                        break;
                    case Z:
                        icp.setInputSource(partitionsz_s[i]);
                        icp.setInputTarget(partitionsz_t[i]);
                        break;
                }
                
                icp.setMaximumIterations(atoi(argv[8]));
                icp.align(*aux_cloud);
                
                // Aplica transformação na source
                Eigen::Matrix4f rt = icp.getFinalTransformation();
                pcl::transformPointCloud(*source, *aux_cloud, rt);
                
                // Verifica RMSE
                double rmse = computeRMSE(target, aux_cloud, numeric_limits<double>::max());
                if (rmse < rmse_min || rmse_min == -1.0) {
                    final_axis = axes_s[p];
                    rmse_min = rmse;
                    rt_min = rt;
                }
                
                if (rmse_min <= rmse_ref) {
                    stop = true;
                    break;
                }
            }
            
            if (stop)
                break;
        }
        
        if (stop)
            break;
    }
    
    // Se 'stop' é True, o critério de parada foi atingido
    // Se for o caso, 'aux' guarda a nuvem alinhada
    if (!stop)
        pcl::transformPointCloud(*source, *aux_cloud, rt_min);
    
    double total_time = (clock() - begin) / (double)CLOCKS_PER_SEC;
    
    // Debug em arquivo dos resultados
    Eigen::Matrix3f r(rt_min.block<3,3>(0, 0));
    Eigen::AngleAxisf angle_axis;
    angle_axis.fromRotationMatrix(r);
    
    ofstream output;
    output.open(argv[9]);
    
    output << "time: " << total_time << endl;
    output << "min partitions: " << kmin << endl;
    output << "max partitions: " << kmax << endl;
    output << "final source" << final_axis <<endl;
    output << "rmse: " << rmse_min << endl;
    output << "rmse reference: " << rmse_ref << endl;
    output << "rt: \n" << rt_min << endl;
    output << "angle: " << (angle_axis.angle()*180.0)/M_PI << endl;
    output << "axis: \n" << angle_axis.axis() << endl;
    
    output.close();
    
    double elem1 = rt_min(0, 0);
    double elem2 = rt_min(1, 1);
    double elem3 = rt_min(2, 2);
    double angle123 = (elem1 + elem2 + elem3 - 1) / 2;
    double axis_angle = acos(angle123) * 180.0 / PI;
    
    cout << "Ration angle:  " << axis_angle << ":  " << endl;
    
    // Visualização
    int sizePoints = 2;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("INPUTS"));
    
    viewer->setBackgroundColor(255, 255, 255);
    viewer->addPointCloud(target, "target");
    viewer->addPointCloud(source, "source");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 255, 0, "source");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, sizePoints, "source");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255, 0, 0,"target");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, sizePoints, "target");
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_align(new pcl::visualization::PCLVisualizer ("ALIGNED"));
    
    viewer_align->setBackgroundColor(255, 255, 255);
    viewer_align->addPointCloud(target, "target");
    viewer_align->addPointCloud(aux_cloud, "aligned");
    viewer_align->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 255, 0, "aligned");
    viewer_align->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, sizePoints, "aligned");
    viewer_align->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 255,"target");
    viewer_align->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, sizePoints, "target");
    
    while(!viewer_align->wasStopped())
        viewer_align->spinOnce();
    
    return 0;
}
