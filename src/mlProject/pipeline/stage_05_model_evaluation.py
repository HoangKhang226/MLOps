from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_evaluation import ModelEvaluation
from src.mlProject import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> STAGE 05: MODEL EVALUATION started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> STAGE 05: MODEL EVALUATION completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
