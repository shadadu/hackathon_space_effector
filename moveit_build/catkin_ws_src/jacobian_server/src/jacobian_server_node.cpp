#include <ros/ros.h>
#include <geometry_msgs/Point.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>

#include <jacobian_server/GetJacobian.h>

class JacobianServer {
public:
  JacobianServer() : nh_("~") {
    // Load robot model from /robot_description (already set by your launch)
    robot_model_loader::RobotModelLoader loader("robot_description");
    kmodel_ = loader.getModel();
    if (!kmodel_) {
      throw std::runtime_error("Failed to load RobotModel from robot_description");
    }

    state_.reset(new moveit::core::RobotState(kmodel_));
    state_->setToDefaultValues();

//    srv_ = nh_.advertiseService("/get_jacobian", &JacobianServer::handle, this);

    ros::NodeHandle nh;
    srv_ = nh.advertiseService("/get_jacobian", &JacobianServer::handle, this);


    ROS_INFO("Jacobian server ready on /get_jacobian");
  }

  bool handle(jacobian_server::GetJacobian::Request &req,
              jacobian_server::GetJacobian::Response &res) {

    if (req.joint_names.size() != req.joint_positions.size()) {
      res.message = "joint_names and joint_positions size mismatch";
      return true;
    }

    const moveit::core::JointModelGroup* jmg = kmodel_->getJointModelGroup(req.group_name);
    if (!jmg) {
      res.message = "Unknown group_name: " + req.group_name;
      return true;
    }

    const moveit::core::LinkModel* link = kmodel_->getLinkModel(req.link_name);
    if (!link) {
      res.message = "Unknown link_name: " + req.link_name;
      return true;
    }

    // Set state from provided joint values (only for the joints in req)
    for (size_t i = 0; i < req.joint_names.size(); ++i) {
      const std::string &jn = req.joint_names[i];
      if (!kmodel_->hasJointModel(jn)) {
        res.message = "Unknown joint: " + jn;
        return true;
      }
      state_->setJointPositions(jn, &req.joint_positions[i]);
    }

    state_->update();

    Eigen::Vector3d ref(req.reference_point.x, req.reference_point.y, req.reference_point.z);

    Eigen::MatrixXd J;
    state_->getJacobian(jmg, link, ref, J);  // 6 x N

    res.rows = (int32_t)J.rows();
    res.cols = (int32_t)J.cols();
    res.jacobian.resize(res.rows * res.cols);

    // Row-major flatten
    for (int r = 0; r < res.rows; ++r) {
      for (int c = 0; c < res.cols; ++c) {
        res.jacobian[r * res.cols + c] = J(r, c);
      }
    }

    res.message = "OK";
    return true;
  }

private:
  ros::NodeHandle nh_;
  ros::ServiceServer srv_;
  moveit::core::RobotModelPtr kmodel_;
  moveit::core::RobotStatePtr state_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "jacobian_server");
  try {
    JacobianServer js;
    ros::spin();
  } catch (const std::exception& e) {
    ROS_ERROR("JacobianServer failed: %s", e.what());
    return 1;
  }
  return 0;
}
