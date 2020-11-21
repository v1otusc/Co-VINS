#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img;
// ros::Publisher pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

/**
 * @brief: 对新来的图像数据进行特征点跟踪，处理和发布
 * @param typedef boost::shared_ptr< ::sensor_msgs::Image const> ImageConstPtr
 */
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
  // 判断是否是第一帧
  if (first_image_flag)
  {
    first_image_flag = false;
    first_image_time = img_msg->header.stamp.toSec();
    last_image_time = img_msg->header.stamp.toSec();
    return;
  }
  // detect unstable camera stream
  // 通过时间间隔判断相机数据是否稳定，有问题则 restart
  if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
  {
    ROS_WARN("image discontinue! reset the feature tracker!");
    first_image_flag = true;
    last_image_time = 0;
    pub_count = 1;
    std_msgs::Bool restart_flag;
    restart_flag.data = true;
    pub_restart.publish(restart_flag);
    return;
  }

  // 更新上一帧图像的时间戳
  last_image_time = img_msg->header.stamp.toSec();

  // frequency control
  // 发布频率控制，保证每秒钟处理的 img_msg 小于 FREQ，默认为频率控制在 10Hz 以内
  // 并不是每读入一帧图像，就要发布此帧的特征点 / 图像
  // 判断间隔内的发布次数
  if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
  {
    PUB_THIS_FRAME = true;
    // reset the frequency control
    // 时间间隔内的发布次数太多十分接近设定频率（大于 9.9 / FREQ）时，更新时间间隔起始时刻，并将数据发布次数置 0
    if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
    {
      first_image_time = img_msg->header.stamp.toSec();
      pub_count = 0;
    }
  }
  else
    PUB_THIS_FRAME = false;

  // 表示一个 OpenCV 图像
  cv_bridge::CvImageConstPtr ptr;
  // 将图像编码 8UC1 转换为 mono8, 单色 8bit
  // TODO: 图像编码格式
  if (img_msg->encoding == "8UC1")
  {
    // ros 图像消息
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    // cv_bridge 的 toCVCopy 函数将 ROS 图像消息转化为 OpenCV 图像，
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  }
  else
    // 默认为 mono8, 单色 8bit
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  cv::Mat show_img = ptr->image;
  TicToc t_r;
  for (int i = 0; i < NUM_OF_CAM; i++)
  {
    ROS_DEBUG("processing camera %d", i);
    // 单目
    if (i != 1 || !STEREO_TRACK)
      // readImage() 函数读取图像数据进行处理
      trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
    // 双目
    else
    {
      // 光太亮或太暗，自适应直方图均衡化处理
      if (EQUALIZE)
      {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
      }
      else
        trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
    }

#if SHOW_UNDISTORTION
    trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
  }

  // TODO: ? 更新全局 id
  for (unsigned int i = 0;; i++)
  {
    bool completed = false;
    for (int j = 0; j < NUM_OF_CAM; j++)
      // 更新 feature 的 id
      if (j != 1 || !STEREO_TRACK)
        completed |= trackerData[j].updateID(i);
    if (!completed)
      break;
  }

  // 将特征点id，矫正后归一化平面的3D点 (x,y,z=1)，像素2D点 (u,v)，像素的速度 (vx,vy)，封装成 sensor_msgs::PointCloudPtr 类型的 feature_points 实例，通过 pub_img 发布
  if (PUB_THIS_FRAME)
  {
    pub_count++;
    sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);

    // sensor_msgs/ChannelFloat32[] channels  #[特征点的ID,像素坐标u,v,速度vx,vy]
    sensor_msgs::ChannelFloat32 id_of_point;
    sensor_msgs::ChannelFloat32 u_of_point;
    sensor_msgs::ChannelFloat32 v_of_point;
    sensor_msgs::ChannelFloat32 velocity_x_of_point;
    sensor_msgs::ChannelFloat32 velocity_y_of_point;

    // std_msgs/Header header #头信息
    feature_points->header = img_msg->header;
    feature_points->header.frame_id = "world";

    vector<set<int>> hash_ids(NUM_OF_CAM);
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
      auto &un_pts = trackerData[i].cur_un_pts;
      auto &cur_pts = trackerData[i].cur_pts;
      auto &ids = trackerData[i].ids;
      auto &pts_velocity = trackerData[i].pts_velocity;

      // 特征点数量，根据 ids.size() 判断
      for (unsigned int j = 0; j < ids.size(); j++)
      {
        // 该特征点被跟踪次数大于 1
        if (trackerData[i].track_cnt[j] > 1)
        {
          int p_id = ids[j];
          hash_ids[i].insert(p_id);
          geometry_msgs::Point32 p;
          p.x = un_pts[j].x;
          p.y = un_pts[j].y;
          p.z = 1;

          feature_points->points.push_back(p);
          id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
          u_of_point.values.push_back(cur_pts[j].x);
          v_of_point.values.push_back(cur_pts[j].y);
          velocity_x_of_point.values.push_back(pts_velocity[j].x);
          velocity_y_of_point.values.push_back(pts_velocity[j].y);
        }
      }
    }
    feature_points->channels.push_back(id_of_point);
    feature_points->channels.push_back(u_of_point);
    feature_points->channels.push_back(v_of_point);
    feature_points->channels.push_back(velocity_x_of_point);
    feature_points->channels.push_back(velocity_y_of_point);
    ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());

    // skip the first image; since no optical speed on frist image
    // 第一帧不发布，因为没有光流速度
    if (!init_pub)
    {
      init_pub = 1;
    }
    else
      pub_img.publish(feature_points);

    // 将图像封装到 cv_bridge::cvtColor 类型的 ptr 实例中，通过 pub_match 发布
    if (SHOW_TRACK)
    {
      ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
      //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
      cv::Mat stereo_img = ptr->image;

      for (int i = 0; i < NUM_OF_CAM; i++)
      {
        cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
        // show_img 灰度图转 RGB（tmp_img）
        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
        for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
        {
          double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
          // 显示跟踪状态，越红越好，越蓝越不好（当前帧中特征点被追踪到的次数越多越好）
          // cv::Scalar() 为设定圆的颜色，规则为 B G R
          cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
          //draw speed line
          /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
          //char name[10];
          //sprintf(name, "%d", trackerData[i].ids[j]);
          //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
      }
      //cv::imshow("vis", stereo_img);
      //cv::waitKey(5);
      //pub_match.publish(ptr->toImageMsg());
    }
  }
  ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
  // ros 初始化以及设置句柄，节点名称为 featrue_tracker
  ros::init(argc, argv, "feature_tracker");
  // 节点句柄，方便对节点进行管理
  ros::NodeHandle n;

  // 设置 logger 的级别，只有级别大于或等于 level 的日志记录消息才会被处理
  // logger_level: DEBUG < INFO < WARN < ERROR < FATAL
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  // 读取 yaml 文件中的参数
  // 其中参数 config_file 在 launch 文件被设置为 $(find feature_tracker)/../config/mynteye/mynteye_config.yaml
  readParameters(n);

  // 读相机内参
  for (int i = 0; i < NUM_OF_CAM; i++)
    trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

  // 如果使用鱼眼相机模型
  if (FISHEYE)
  {
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
      // config/fisheye_mask.jpg
      // cv::imread() 读入图像单通道，即灰度图
      trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
      // data: Mat对象中的一个指针类型的成员变量，指向存放 Mat 矩阵数据的首地址(uchar* data)，是一个地址的指针
      if (!trackerData[i].fisheye_mask.data)
      {
        ROS_INFO("load mask fail");
        ROS_BREAK();
      }
      else
        ROS_INFO("load mask success");
    }
  }

  // 订阅相机发布的图像 topic
  ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
  // 跟踪的特征点信息，由 /vins_estimator 订阅并进行优化
  pub_img = n.advertise<sensor_msgs::PointCloud>("feature_tracker/feature", 1000);
  // 跟踪的特征点图像，给 rviz 和调试用
  // pub_match = n.advertise<sensor_msgs::Image>("feature_tracker/feature_img",1000);
  // 判断特征跟踪模块是否出错，若有问题则进行复位，由 /vins_estimator 订阅
  pub_restart = n.advertise<std_msgs::Bool>("feature_tracker/restart", 1000);
  /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
  //
  ros::spin();
  return 0;
}

// new points velocity is 0, pub or not?
// track cnt > 1 pub?