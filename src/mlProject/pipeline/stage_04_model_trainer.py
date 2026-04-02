from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_trainer import ModelTrainer
from src.mlProject import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.initiate_model_trainer()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> STAGE 04: MODEL TRAINER started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> STAGE 04: MODEL TRAINER completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
