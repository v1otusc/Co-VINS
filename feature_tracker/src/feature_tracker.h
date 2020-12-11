#pragma once

#include <execinfo.h>

#include <csignal>
#include <cstdio>
#include <iostream>
#include <queue>
using namespace std;

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
using namespace Eigen;

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
using namespace camodocal;

#include "parameters.h"
#include "tic_toc.h"

bool inBorder(const cv::Point2f &pt);
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker {
 public:
  FeatureTracker();

  void readImage(const cv::Mat &_img, double _cur_time);

  void setMask();

  void addPoints();

  bool updateID(unsigned int i);

  void readIntrinsicParameter(const string &calib_file);

  void showUndistortion(const string &name);

  void rejectWithF();

  void undistortedPoints();

  cv::Mat mask;
  // 鱼眼相机 mask，用来去除边缘噪声点
  cv::Mat fisheye_mask;
  // prev_img // TODO: ? 是上一帧发布的图像数据
  // cur_img 表示前一帧，是光流跟踪前一帧的图像数据
  // forw_img 表示当前帧，是光流跟踪后一帧的图像数据
  cv::Mat prev_img, cur_img, forw_img;
  // 每一帧中通过 cv::goodFeaturesToTrack() 新提取到的特征点
  vector<cv::Point2f> n_pts;
  // 与 prev_img cur_img forw_img 对应的图像特征点
  vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
  //
  vector<cv::Point2f> prev_un_pts, cur_un_pts;
  //
  vector<cv::Point2f> pts_velocity;
  // 能够被跟踪到的特征点的 id
  vector<int> ids;
  // 当前帧 forw_img 中每个特征点被追踪到的次数
  vector<int> track_cnt;
  map<int, cv::Point2f> cur_un_pts_map;
  map<int, cv::Point2f> prev_un_pts_map;
  // 相机模型
  camodocal::CameraPtr m_camera;
  double cur_time;
  double prev_time;

  // 类中的静态成员变量，不属于某个类实例对象，用来作为特征点
  // id，每检测到一个新的特征点，就将 n_id 作为该特征点的 id，然后 n_id 加 1
  static int n_id;
};
