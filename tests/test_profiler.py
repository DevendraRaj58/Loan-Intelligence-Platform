from src.ingestion.data_ingestion import DataIngestion
from src.utils.profiler import DatasetProfiler

ingestion = DataIngestion()

df = ingestion.load_csv("home_credit/application_train.csv")

profiler = DatasetProfiler(df)

profiler.profile()