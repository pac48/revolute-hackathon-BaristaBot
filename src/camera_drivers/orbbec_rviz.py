"""RViz for the Orbbec Gemini 305 (config/orbbec.rviz); see rviz.py."""

from src.camera_drivers import orbbec_rviz_config, rviz

if __name__ == "__main__":
    rviz.main(orbbec_rviz_config)
