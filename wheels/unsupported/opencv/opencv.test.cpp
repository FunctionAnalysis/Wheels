#include <gtest/gtest.h>

#include "../../src/constants.hpp"
#include "../../src/remap.hpp"
#include "../../src/tensor.hpp"
#include "../../src/time.hpp"
#include "../../src/upgrade.hpp"

#include "../soil/image.hpp"

#include "opencv.hpp"

using namespace wheels;

TEST(opencv, opencv) {
  matx m1(make_shape(50, 50));

  m1 = ones(100, 100);

  matx_<vec3> mmm(make_shape(50, 50));

  //auto im = imread(wheels_data_dir_str "/wheels.jpg").eval();

  cv::Mat_<cv::Vec<uint8_t, 3>> cvim = cv::imread(wheels_data_dir_str "/wheels.jpg");
  image3u8 im(make_shape(cvim.rows, cvim.cols));
  for (int i = 0; i < cvim.rows; i++) {
	  for (int j = 0; j < cvim.cols; j++) {
		  for (int k = 0; k < 3; k++) {
			  im(i, j)[k] = cvim(i, j)[k];
		  }
	  }
  }

  //auto im = load_image<3>(wheels_data_dir_str "/wheels.jpg");

  /* auto nim = remap(upgrade_all(im), make_shape(800, 200, 3), [](auto y, auto
     x,
                                                                 auto c) {
                return vec3(x / 2.0 + 300.0, y / 3.0, c / 2.0);
              }).eval();*/
}
