import argparse
import json
import time

import sys
sys.path.append('/')

import logging
from tmp_utils import set_logger
from remote.GLM3 import ChatGLMLLM
from  remote.ChatGPT import ChatGPTLLM
from JudgeAgent import JudgeAgent


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--i', type=str, help='input JSON file with multiple cases')
    parser.add_argument('--o', type=str, help='output JSON file for results')
    parser.add_argument('--r', action='store_true', help='resume from the last saved state')


    args = parser.parse_args()

    # 设置日志
    logger = set_logger("gpt_tmp.log")

    # 初始化模型和参数
    glm3_config_path = '../remote/configs/glm3.json'
    gpt35_config_path = '../remote/configs/gpt35.json'
    glm3 = ChatGLMLLM(glm3_config_path)
    gpt35 = ChatGPTLLM(gpt35_config_path)
    task_name = "Choose Correcr Answer And Give Analyze Based On Context and Question"
    question = "Question"
    language = "Chinese"
    result_pattern = {
        "Answer": "str_",
        "Analysis": "str_"
    }
    more_guidance = ['selecting the answer depends on context,not guesswork.']
    in_context_examples = [{
            "Input": {
                "Question": "莎士比亚是英国文学史上最伟大的戏剧作家之一，他的作品被翻译成多种语言并在世界范围内广泛上演。"
                            "其中，他的悲剧《哈姆雷特》被认为是他最杰出的作品之一。谁是《哈姆雷特》的作者?A.哈罗德·品钦B.查尔斯·狄更斯C.威廉·莎士比亚D.简·奥斯汀",
            },
            "Output": {
                "Answer": "A. 2",
                "Analysis": "小红购买的商品价值为280元，因此可以获得2次抽奖机会（每满100元获得1次）。所以小红在这次抽奖活动中获得了2次抽奖机会。"
            }
        }]  # 示例数据

    # 初始化JudgeAgent
    judge_agent = JudgeAgent(logger, gpt35, task_name, result_pattern,
                             language=language, question=question,
                             more_guidance=more_guidance, in_context_examples=in_context_examples)

    # 读取输入数据
    try:
        with open(args.i, 'r', encoding='utf-8') as f:
            multiple_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    except json.JSONDecodeError:
        logger.error(f"Input file is not a valid JSON: {args.input}")
        sys.exit(1)


    results = []
    tmps = []
    # 断点续传逻辑
    processed_data = []
    if args.r:
        try:
            with open('tmp.json', 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
                tmps = processed_data

            with open(args.o, 'r', encoding='utf-8') as f:
                results = json.load(f)  # 读取已有的结果
        except FileNotFoundError:
            logger.info(f"No previous output file found, starting from scratch.")
        except json.JSONDecodeError:
            logger.error(f"Output file is not a valid JSON: {args.output}")
            sys.exit(1)

    # 处理数据

    for data in multiple_data:
        if data in processed_data:
            logger.info(f"Skipping already processed data: {data}")
            continue
        try:
            result = judge_agent.judge_a_case(data)

            results.append(result[-1])
            logger.info(f'Had process data:{data}')

            tmps.append(data)
            # 保存已经处理过的数据
            with open('tmp.json', 'w', encoding='utf-8') as f:
                json.dump(tmps, f, ensure_ascii=False, indent=4)
            # 保存已经处理出的结果
            with open(args.o, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"Error processing data: {data}. Error: {e}")
            sys.exit(1)



if __name__ == '__main__':

    start_time = time.time()  # 获取开始时间
    main()
    end_time = time.time()  # 获取结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"程序运行时间: {elapsed_time} 秒")
