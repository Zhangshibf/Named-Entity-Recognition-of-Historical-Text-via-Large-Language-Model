from src.datasetloader import Dataset_loader
from src.llm_client import llm_client
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline with LLM client and dataset.")
    parser.add_argument("dataset_name", type=str,help="Name of the dataset")
    parser.add_argument("language", type=str,help="Language of the dataset")
    parser.add_argument("split", type=str, default="test", help="Split of the dataset to use. default test")
    parser.add_argument("temperature",type = float,help="temperature of LLM")
    parser.add_argument("task", type=str,  help="Task name")
    parser.add_argument("prompt", type=str, help="prompt method")
    parser.add_argument("key", type=str, help="API or system key")
    parser.add_argument("system", type=str,  help="LLM system name")
    parser.add_argument("subfolder",type=str,help="Use parameter such as round1, round2, ... This is going to be the subfolder in prediction folder")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state")

    args = parser.parse_args()


    client = llm_client()
    dataset = Dataset_loader(args.dataset_name, args.language, args.split)


    assert args.prompt in ["b","r_m1","r_m3","r_m5","s_overlap_1","s_overlap_3","s_overlap_5","s_embedding_1","s_embedding_3","s_embedding_5"] #baseline, random example, multiple random examples, similar example, multiple similar example
    client.run(args.prompt,dataset, args.task, args.system, args.key,args.temperature,args.subfolder,args.resume)