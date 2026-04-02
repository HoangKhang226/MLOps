from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.pipeline.prediction import PredictionPipeline
from src.mlProject import logger
import os

STAGE_NAME = "Serving & UI Stage"


class PredictionServicePipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Since this is a serving stage, we initiate the FastAPI and Streamlit apps if needed,
        or just confirm all prediction components are ready.
        """
        try:
            logger.info("Initializing Stage 06: Prediction Service...")
            # Check if artifacts exist
            if not os.path.exists("artifacts/model_trainer/model.joblib"):
                logger.error(
                    "Model not found in artifacts! Run Model Trainer (Stage 04) first."
                )
                return

            logger.info("Serving & Prediction components are READY.")
            logger.info("Use 'python app.py' to start the FastAPI server.")
            logger.info("Use 'streamlit run streamlit_app.py' to start the UI.")

        except Exception as e:
            logger.exception(f"Error in Stage 06: {e}")
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PredictionServicePipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
