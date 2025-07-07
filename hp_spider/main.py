import boto3
import os
import sys
#import json
from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.helpers import models_tokens

"""
path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
os.chdir(path)
sys.path.append(path)
"""
#from lib import BedrockConfig


class BedrockConfig:

    BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    EMBEDDED_MODEL = "cohere.embed-multilingual-v3"
    TEMPERATURE = 0.40
    MAX_TOKENS = 4096

    def __init__(self):
        pass


class BedrockLlm:
    def __init__(self, config):
        session = boto3.Session()
        client = session.client("bedrock-runtime")

        
        self.graph_config = {
            "llm": {
                "client": client,
                "model": f"bedrock/{config.BEDROCK_MODEL}",
                "temperature": config.TEMPERATURE,
                "max_tokens": config.MAX_TOKENS,
                "model_kwargs": {
                }
            },
            "embeddings": {
                "client": client,
                "model": f"bedrock/{config.EMBEDDED_MODEL}",
                "temperature": 0.0,
                "model_kwargs": {
                }
            }
        }

    def create(self, prompt, source):
        self.graph = SmartScraperGraph(
            prompt=prompt,
            source=source,
            config=self.graph_config
        )

    def run(self):
        result = self.graph.run()
        print(result)
        # j_result = JSON.parse(result)
        return result # j_result["content"]


# Lambda handler function
def lambda_handler(event, context):

    prompt = event["prompt"]
    source = event["source"]

    llm = BedrockLlm(BedrockConfig)
    llm.create(prompt, source)
    content = llm.run()

    return {
        "statusCode": 200,
        "body": content,
    }
