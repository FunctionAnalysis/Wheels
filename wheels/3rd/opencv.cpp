#include "opencv.hpp"

namespace wheels {

namespace details {
cv::Mat _imread(const filesystem::path &path) {
  std::ifstream file(path, std::iostream::binary);
  if (!file.good()) {
    return cv::Mat();
  }
  file.exceptions(std::ifstream::badbit | std::ifstream::failbit |
                  std::ifstream::eofbit);
  file.seekg(0, std::ios::end);
  std::streampos length(file.tellg());
  std::vector<char> buffer(static_cast<std::size_t>(length));
  if (static_cast<std::size_t>(length) == 0) {
    return cv::Mat();
  }
  file.seekg(0, std::ios::beg);
  try {
    file.read(buffer.data(), static_cast<std::size_t>(length));
  } catch (...) {
    return cv::Mat();
  }
  file.close();
  cv::Mat image = cv::imdecode(buffer, CV_LOAD_IMAGE_COLOR);
  return image;
}

bool _imwrite(const filesystem::path &path, const cv::Mat &mat) {
  std::basic_ofstream<unsigned char, std::char_traits<unsigned char>> file(
      path, std::iostream::binary);
  if (!file.good()) {
    return false;
  }
  file.seekp(0, std::ios::beg);
  file.exceptions(std::ifstream::badbit);
  std::vector<unsigned char> buffer;
  buffer.reserve(mat.rows * mat.cols * mat.channels() * mat.elemSize());
  if (!cv::imencode(".jpg", mat, buffer)) {
    return false;
  }
  if (buffer.empty()) {
    return false;
  }
  try {
    file.write(buffer.data(), buffer.size());
    file.flush();
  } catch (...) {
    return false;
  }
  file.close();
  return true;
}

// _vdread
std::vector<cv::Mat> _vdread(const filesystem::path &filepath) {
  std::vector<cv::Mat> frames;
  cv::VideoCapture cap(filepath.string());
  if (!cap.isOpened()) {
    return frames;
  }
  auto n = static_cast<size_t>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  frames.reserve(n);
  cv::Mat cur_fram;
  while (cap.read(cur_fram)) {
    frames.push_back(cur_fram.clone());
  }
  return frames;
}
// _vdwrite
bool _vdwrite(const filesystem::path &path,
              const std::vector<cv::Mat> &frames) {
  throw std::runtime_error("not implemented yet");
  return false;
}
}
}
