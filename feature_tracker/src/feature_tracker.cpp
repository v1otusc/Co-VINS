#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
  const int BORDER_SIZE = 1;
  //cvRound() 返回跟参数最接近的整数值，四舍五入
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
  if (FISHEYE)
    mask = fisheye_mask.clone();
  else
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned int i = 0; i < forw_pts.size(); i++)
    cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
    return a.first > b.first;
  });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (auto &it : cnt_pts_id)
  {
    if (mask.at<uchar>(it.second.first) == 255)
    {
      forw_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
    }
  }
}

void FeatureTracker::addPoints()
{
  for (auto &p : n_pts)
  {
    forw_pts.push_back(p);
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

// 对图像使用光流法进行特征点跟踪
/**
 * @brief: 
 * @param {*}
 * @return {*}
 */
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
  cv::Mat img;
  TicToc t_r;
  cur_time = _cur_time;

  // 若控制 config/ 文件中的参数 EQUALIZE 为 1，表示太亮或者太暗，调用 cv::creatCLAHE() 对输入图像做自适应直方图均衡
  // TODO: 直方图均衡的原理
  if (EQUALIZE)
  {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    TicToc t_c;
    clahe->apply(_img, img);
    ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
  }
  else
    img = _img;

  if (forw_img.empty())
  {
    // 如果当前帧的图像数据 forw_img 为空，说明当前是第一次读入图像数据
    // 将读入的图像赋给当前帧 forw_img，同时还赋给 prev_img cur_img
    prev_img = cur_img = forw_img = img;
  }
  else
  {
    // 否则，说明之前就已经有图像读入
    // 所以只需要更新当前帧 forw_img 的数据
    forw_img = img;
  }

  // 此时 forw_pts 还保存的是上一帧图像中的特征点，所以把它清除
  forw_pts.clear();

  // 上一帧有特征点
  if (cur_pts.size() > 0)
  {
    TicToc t_o;
    vector<uchar> status;
    vector<float> err;

    // TODO: 光流跟踪的原理
    // 调用 cv::calcOpticalFlowPyrLK() 对前一帧的特征点 cur_pts 进行 LK 金字塔光流跟踪，得到 forw_pts
    // status 标记了从前一帧 cur_img 到 forw_img 特征点的跟踪状态，无法被追踪到的点标记为 0
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++)
      // 将当前帧跟踪的位于图像边界外的点标记为 0
      if (status[i] && !inBorder(forw_pts[i]))
        status[i] = 0;
    // 不仅要从当前帧数据 forw_pts 中剔除，而且还要从 cur_un_pts prev_pts 和 cur_pts 中剔除
    // prev_pts 和 cur_pts 中的特征点是一一对应的
    // 记录特征点 id 的 ids，和记录特征点被跟踪次数的 track_cnt 也要剔除
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
  }

  for (auto &n : track_cnt)
    n++;

  // 需要发布特征点
  if (PUB_THIS_FRAME)
  {
    // 通过基本矩阵剔除 outliers
    rejectWithF();
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    // setMask() 保证相邻的特征点之间要相隔30个像素
    setMask();
    ROS_DEBUG("set mask costs %fms", t_m.toc());

    // 寻找新的特征点 goodFeaturesToTrack()
    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0)
    {
      if (mask.empty())
        cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1)
        cout << "mask type wrong " << endl;
      if (mask.size() != forw_img.size())
        cout << "wrong size " << endl;
      
      /** 
       *void cv::goodFeaturesToTrack(            在mask中不为0的区域检测新的特征点
       *   InputArray  image,                    输入图像
       *   OutputArray corners,                  存放检测到的角点的 vector
       *   int         maxCorners,               返回的角点的数量的最大值
       *   double      qualityLevel,             角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
       *   double      minDistance,              返回角点之间欧式距离的最小值
       *   InputArray  mask = noArray(),         和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
       *   int         blockSize = 3,            计算协方差矩阵时的窗口大小
       *   bool        useHarrisDetector = false,指示是否使用Harris角点检测，如不指定则使用 shi-tomasi 算法
       *   double      k = 0.04                  Harris角点检测需要的k值
       *)   
       */
      cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
    }
    else
      n_pts.clear();
    ROS_DEBUG("detect feature costs: %fms", t_t.toc());

    ROS_DEBUG("add feature begins");
    TicToc t_a;
    // addPoints() 向 forw_pts 添加新的追踪点
    addPoints();
    ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
  }

  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  cur_img = forw_img;
  cur_pts = forw_pts;

  // 根据不同的相机模型去畸变矫正和转换到归一化坐标系上，计算速度
  undistortedPoints();
  prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
  // 当前帧追踪上的特征点数量足够多
  if (forw_pts.size() >= 8)
  {
    ROS_DEBUG("FM ransac begins");
    TicToc t_f;
    // 
    vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
      Eigen::Vector3d tmp_p;
      m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
    int size_a = cur_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
    ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

bool FeatureTracker::updateID(unsigned int i)
{
  if (i < ids.size())
  {
    if (ids[i] == -1)
      ids[i] = n_id++;
    return true;
  }
  else
    return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
  ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
  cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < COL; i++)
    for (int j = 0; j < ROW; j++)
    {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
      //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }
  for (int i = 0; i < int(undistortedp.size()); i++)
  {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
    pp.at<float>(2, 0) = 1.0;
    //cout << trackerData[0].K << endl;
    //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
    {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    }
    else
    {
      //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  cv::imshow(name, undistortedImg);
  cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
  cur_un_pts.clear();
  cur_un_pts_map.clear();
  //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
  for (unsigned int i = 0; i < cur_pts.size(); i++)
  {
    Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    m_camera->liftProjective(a, b);
    cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
    //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
  }
  // caculate points velocity
  if (!prev_un_pts_map.empty())
  {
    double dt = cur_time - prev_time;
    pts_velocity.clear();
    for (unsigned int i = 0; i < cur_un_pts.size(); i++)
    {
      if (ids[i] != -1)
      {
        std::map<int, cv::Point2f>::iterator it;
        it = prev_un_pts_map.find(ids[i]);
        if (it != prev_un_pts_map.end())
        {
          double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.push_back(cv::Point2f(v_x, v_y));
        }
        else
          pts_velocity.push_back(cv::Point2f(0, 0));
      }
      else
      {
        pts_velocity.push_back(cv::Point2f(0, 0));
      }
    }
  }
  else
  {
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}
