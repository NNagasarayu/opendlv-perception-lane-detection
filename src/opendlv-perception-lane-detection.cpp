/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>
#include <string> 
#include <cmath> 

bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, uint32_t index, cv::Mat& A);

std::vector<int32_t> parseConesMsg(std::string s);

int32_t main(int32_t argc, char **argv) {
  int32_t retCode{1};
  auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if ( (0 == commandlineArguments.count("cid")) ||
       (0 == commandlineArguments.count("freq")) ||
       (0 == commandlineArguments.count("height")) ||
       (0 == commandlineArguments.count("width")) ||
       (0 == commandlineArguments.count("dist-to-curve"))  ) {
    std::cerr << argv[0] << " takes it inputs from a UDP conference." << std::endl;
    std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --freq=<frequency at which message is sent> --height=<height of coorinate system> --width=<width of coordinate system> --dist-to-curve=<distance to cone wall to classify as curve> [--cones-intersect] [--max-cones] [--task3] [--verbose]" << std::endl;
    std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
    std::cerr << "         --freq:   The frequency for seneding new lane responses" << std::endl;
    std::cerr << "         --height:   The height of the image used when finding the cone" << std::endl;
    std::cerr << "         --width:   The width of the image used when finding the cone" << std::endl;
    std::cerr << "         --dist-to-curve:   The distance in pixels used to find wall infront" << std::endl;
    std::cerr << "         --cones-intersect:   The number of red cones above which we classify the situation as a intersection" << std::endl;
    std::cerr << "         --max-cones:  The maximum number of cones used in the linear regression to find the lanes" << std::endl;
    std::cerr << "Example: " << argv[0] << " --cid=111 --freq=10 --heigt=240 --width=1280 --dist-to-curve=100 --verbose" << std::endl;
  } else {
    const uint32_t freq{static_cast<uint32_t>(
      std::stoi(commandlineArguments["freq"]))};
    const uint32_t height{static_cast<uint32_t>(
      std::stoi(commandlineArguments["height"]))};
    const uint32_t width{static_cast<uint32_t>(
      std::stoi(commandlineArguments["width"]))};
    const uint32_t conesForIntersect = (commandlineArguments.count(
      "cones-intersect") != 0) ? static_cast<uint32_t>(std::stoi(
      commandlineArguments["cones-intersect"])) : 2;
    const uint32_t maxCones = (commandlineArguments.count(
      "max-cones") != 0) ? static_cast<uint32_t>(std::stoi(
      commandlineArguments["max-cones"])) : 0;
    double const curveWallDist{std::stod(commandlineArguments["dist-to-curve"])};
    const bool VERBOSE{commandlineArguments.count("verbose") != 0};
    const bool task3{commandlineArguments.count("task3") != 0};
    // Interface to a running OpenDaVINCI session; here, 
    // you can send and receive messages.
    cluon::OD4Session od4{static_cast<uint16_t>(
      std::stoi(commandlineArguments["cid"]))};
    
    std::mutex conesMutex;
    std::vector<std::string> allXCoordsCones(3,
        std::string());
    std::vector<std::string> allYCoordsCones(3,
      std::string());
    std::vector<uint32_t> keepOldCount(3, 0);
    uint32_t coneCount{0};
    bool blueRight{true};
    
    auto onCone = [&conesMutex, &allXCoordsCones, &allYCoordsCones, &keepOldCount, &VERBOSE](cluon::data::Envelope &&env){
        auto senderStamp = env.senderStamp();
        opendlv::logic::perception::ConePosition cp =
          cluon::extractMessage<opendlv::logic::perception::ConePosition>
          (std::move(env));
        
        // bool if to keep old message when no cones in new message
        bool keepOld = (cp.x().size()==0 && senderStamp<3 &&
          allXCoordsCones[senderStamp].size()>0 && keepOldCount[senderStamp]<3);
        
        if (!keepOld) {
          std::lock_guard<std::mutex> lock(conesMutex);
          if (senderStamp==0) {
            allXCoordsCones[cp.color()] = cp.x();
            allYCoordsCones[cp.color()] = cp.y();
            keepOldCount[senderStamp]=0;
          } else if (senderStamp==1) {
            allXCoordsCones[cp.color()] = cp.x();
            allYCoordsCones[cp.color()] = cp.y();
            keepOldCount[senderStamp]=0;
          } else if (senderStamp==2) {
            allXCoordsCones[cp.color()] = cp.x();
            allYCoordsCones[cp.color()] = cp.y();
            keepOldCount[senderStamp]=0;
          }
        } else if (senderStamp<3) {
          keepOldCount[senderStamp]++;
        }
        if (VERBOSE) {
          std::cout << "recieved cones : "<< cp.color() << "\nx=" << 
            allXCoordsCones[cp.color()]<< "\ny=" << allYCoordsCones[cp.color()]   
            << "\n keeptOld:" << keepOld << std::endl;
        }
    };
    
    auto atFrequency{[&conesMutex, &allXCoordsCones, &allYCoordsCones, &width,
      &height, &curveWallDist, &coneCount, &blueRight, &conesForIntersect, &od4, &VERBOSE, &task3, &maxCones]() -> bool
    {
      //For plotting if verbose
      cv::Mat image; 
      cv::Point offset;
      if (VERBOSE){
        image = cv::Mat::zeros(height, width, CV_8UC3);
        image.setTo(cv::Scalar(100, 0, 0));
      }
      int32_t closePointX = width/2-80;
      
      //Part I convert string to coordinates
      bool intersection {false};
      std::vector<std::vector<cv::Point>> points = std::vector<std::vector<
        cv::Point>>(3, std::vector<cv::Point>()); //blue, yellow, red
      {
        for (uint32_t i=0; i<3; i++) {
          std::lock_guard<std::mutex> lock(conesMutex);
          std::vector<int32_t> conesX = parseConesMsg(allXCoordsCones[i]);
          std::vector<int32_t> conesY = parseConesMsg(allYCoordsCones[i]);
          for (uint32_t j=0; j < conesX.size(); j++) {
            points[i].push_back(cv::Point(conesX[j], conesY[j]));
          }
          
        }
           
        if (points[2].size() > conesForIntersect && task3) {
          intersection = true;
        }         
        
        //Adds red cones to either blue or yellow depending on x value
        if (!intersection) {
          if (points[2].size()==1) {
            if ((points[2][0].x>0)==blueRight) {
              points[0].push_back(points[2][0]);
            } else {
              points[1].push_back(points[2][0]);
            }
          } else if (points[2].size() ==2) {
            int32_t leftIndex;
            if (points[2][0].x<points[2][1].x) {
              leftIndex = 0; 
            } else {
              leftIndex = 1;
            }
            if (blueRight) {
              points[0].push_back(points[2][1-leftIndex]);
              points[1].push_back(points[2][leftIndex]);
            } else {
              points[0].push_back(points[2][leftIndex]);
              points[1].push_back(points[2][1-leftIndex]);
            }
          }
        } else {                   
          //task3 code
          for (uint32_t i = 0; i< points[2].size(); i++) {
            if ((points[2][i].x > 0) == blueRight) {
              points[0].push_back(points[2][i]);
            } else {
              points[1].push_back(points[2][i]);
            }
          }
        }
    
        // In the begining looks for three consective instances where the closest
        // yellow cone is on the same side of the closest blue cone all three 
        // instances. Determins which side is which color.
        if (coneCount < 3 && points[0].size() > 0 && points[1].size() > 0){
          int minYBlue{height};
          int xBlue;
          int minYYellow{height};
          int xYellow;
          for (auto point: points[0]) {
            if (point.y < minYBlue) {
              minYBlue = point.y;
              xBlue = point.x;
            } 
          }
          for (auto point: points[1]) {
            if (point.y < minYYellow) {
              minYBlue = point.y;
              xYellow = point.x;
            } 
          }
          if ((xYellow<xBlue)!=blueRight){
            blueRight = (xYellow<xBlue);
            coneCount = 0;
          } else {
            coneCount++;
          }
        }
        
        // Make sure that we only use the maxCones closest cones (measured in y)
        if (maxCones>0 && points[0].size() > maxCones) {
          std::sort(points[0].begin(), points[0].end(), 
            [](const cv::Point &a, const cv::Point &b) {
              return (a.y<b.y);
            });
          points[0].resize(maxCones);
        }
        if (maxCones>0 && points[1].size() > maxCones) {
          std::sort(points[1].begin(), points[1].end(), 
            [](const cv::Point &a, const cv::Point &b) {
              return (a.y<b.y);
            });
          points[1].resize(maxCones);
        }
        
        //Add point close to wheel for either side
        points[0].push_back(cv::Point((blueRight)?closePointX:-closePointX, 1));
        points[1].push_back(cv::Point((blueRight)?-closePointX:closePointX, 1));
      }
      
      // Part II: Find the poly fit of degree deg 
      uint32_t const deg{1}; // 1 linear, 2 quad,..
      cv::Mat A = cv::Mat::zeros(deg + 1, 2, CV_64FC1); // blue line and 
      // yellow line
      {
        for (uint32_t i=0; i<2; i++) {
          
          polynomial_curve_fit(points[i], deg, i, A);
          
          if (VERBOSE) {
            std::vector<cv::Point> plotPoints;
            cv::Scalar color = (i==0) ? cv::Scalar(255, 0, 0) :
              cv::Scalar(0, 255, 255);
            
            for (auto point: points[i]) {
              plotPoints.push_back(cv::Point(point.x+width/2, height-point.y));
            }
            
            
            for (uint32_t j = 0; j < points[i].size(); j++) {
              cv::circle(image, plotPoints[j], 5, color, 2, 8, 0);
            }
            std::vector<cv::Point> points_fitted;
            for (int x = -image.cols/2; x < image.cols/2; x++) {
              double y{0.0};
              for (uint32_t d=0; d <= deg; d++) {
                y += A.at<double>(d, i) * std::pow(x, d);
              }
              points_fitted.push_back(cv::Point(x+width/2, height-y));
            }
            
            cv::polylines(image, points_fitted, false, color, 1, 
              8, 0);
          }
        }
      }
      
      //Part III: Find aim point (Only works for deg 1)
      {
        double yBlue{0.0};
        double yYellow{0.0};
        bool noBlue{false};
        bool noYellow{false};
        
        //finds the y coord of x=0
        if (points[0].size() > 1) {
          for (uint32_t d=0; d <= deg; d++) {
            yBlue += A.at<double>(d, 0) * std::pow(0, d);
          }
        } else {
          yBlue = -0.1;
          noBlue = true;
        }
        if (points[1].size() > 1) {
          for (uint32_t d=0; d <= deg; d++) {
            yYellow += A.at<double>(d, 1) * std::pow(0, d);
          }
        } else {
          yYellow = -0.1;
          noYellow = true;
        }
        
        double x{0.0};
        double y{curveWallDist};
        int color{-1};
        
        if (yBlue > 0 && yBlue < y) {
          color = 0;
          y = yBlue;
        } else if (yYellow > 0 && yYellow < y) {
          color = 1;
          y = yYellow;
        }
        
        if (noYellow && noBlue) {
          // bad
          y = -100.0;
        } else {
          //invert (only deg = 1)
          double xBlue{(y-A.at<double>(0, 0))/A.at<double>(1, 0)};
          double xYellow{(y-A.at<double>(0, 1))/A.at<double>(1, 1)};
          if (noBlue) {
            xBlue = (blueRight) ? closePointX : -closePointX;
          } else if (noYellow) {
            xYellow = (blueRight) ? -closePointX : closePointX;
          }
                  
          if (color == -1) {
            x = (xBlue + xYellow) / 2;
          } else if (color == 0) {
            // Blue wall in front turn to yellow
            x = xYellow;
          } else if (color == 1) {
            // Yellow wall in front turn to blue
            x = xBlue;
          }
          if (x < -closePointX){
            x = -closePointX;
          } else if ( x > closePointX){
            x = closePointX;
          }
        }
        
        double yawError = std::atan2(-x, y); //y is straight ahead
        double distance = y;
        
        opendlv::logic::action::AimPoint aim;
        aim.azimuthAngle(static_cast<float>(yawError));
        aim.zenithAngle(0.0f);
        aim.distance(static_cast<float>(distance));
        cluon::data::TimeStamp sampleTime;
        if (intersection) {
          od4.send(aim, sampleTime, 1);
        } else {
          od4.send(aim, sampleTime, 0);
        }
        
        if (VERBOSE) {
          std::cout << "x=" << x << ", y=" << y << std::endl;
          std::cout << "yawError=" << yawError << ", distance=" << distance 
            << ", intersection="<<intersection << std::endl;
          cv::circle(image, cv::Point(static_cast<int>(x)+width/2,
            height-static_cast<int>(y)), 5, cv::Scalar(0, 0, 255), 2, 8, 0);
        }
      }
      
      if (VERBOSE) {
        cv::imshow("image", image);
        cv::waitKey(1);
      }
      return true;
    }};
    // Finally, we register our lambda for the message identifier for the 
    // conesMsg
    od4.dataTrigger(opendlv::logic::perception::ConePosition::ID(), onCone);
    
    // Register the time trigger, spawning a thread that blocks execution 
    // until CTRL-C is pressed
    od4.timeTrigger(freq, atFrequency);
    retCode = 0;
  }
  return retCode;
}


bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, uint32_t index, cv::Mat& A)
{
  //Number of key points
  int N = key_point.size();

  //Construct matrix X
  cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
  for (int i = 0; i < n + 1; i++)
  {
    for (int j = 0; j < n + 1; j++)
    {
      for (int k = 0; k < N; k++)
      {
        X.at<double>(i, j) = X.at<double>(i, j) +
            std::pow(key_point[k].x, i + j);
      }
    }
  }

  //Construct matrix Y
  cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
  for (int i = 0; i < n + 1; i++)
  {
    for (int k = 0; k < N; k++)
    {
      Y.at<double>(i, 0) = Y.at<double>(i, 0) +
          std::pow(key_point[k].x, i) * key_point[k].y;
    }
  }
  
  //Solve matrix A
  cv::solve(X, Y, A.col(index), cv::DECOMP_LU);
  return true;
}

std::vector<int32_t> parseConesMsg(std::string s)
{
  std::vector<int32_t> strings;
  std::string delimiter = ",";
  
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    strings.push_back(stoi(token));
    s.erase(0, pos + delimiter.length());
  }
  if (s.length() > 0 ){
    strings.push_back(stoi(s));
  }
  return strings;
}


