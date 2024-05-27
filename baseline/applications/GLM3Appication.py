import argparse
import json

from applications.prompts import EN_TEXT_EVAL_GENERAL_PROMPT_PATTERN, EN_TEXT_EVAL_METRICS
from applications.tmp_utils import set_logger
from remote import RemoteLLMs
from remote.GLM3 import ChatGLMLLM


class JudgeAgent:
    def __init__(self, logger, llm_model: RemoteLLMs, task_name: str,metrics: dict,
                 language, background, question, options, in_context_examples=[],
                 more_guidance=[], more_task_definition=[]):
        """

        :param logger:
        :param llm_model: 给定一个RemoteLLMs的实例化对象
        :param task_name:  需要指定该任务的语言
        :param language:  数据的语言
        :param metrics: 返回结果模板
        :param background: 背景材料
        :param question: 问题
        :param options: 选项
        :param in_context_examples: 如果需要给定In Context的例子，给对应的数组，每个例子一个Dict
        :param more_guidance: 通过的数组的形式提供补助
        :param more_task_definition:  通过数组的形式提供更多的任务定义补充
        """
        self.logger = logger
        self.llm_model = llm_model
        self.prompt_pattern = EN_TEXT_EVAL_GENERAL_PROMPT_PATTERN
        self.result_pattern = metrics

        # 处理额外的输入
        more_task_definition = '\n'.join(more_task_definition)

        # 评价的指标
        metric_dict = dict()
        data_type = None
        for metric, score_format in metrics.items():
            metric_dict[metric] = "[Your Result]"
            items = score_format.split('_')

            if data_type is None:
                data_type = items[0]
            else:
                assert data_type == items[0]

        output_pattern = json.dumps(metric_dict, ensure_ascii=False, indent=4)

        # In-Context Examples 的设置
        if len(in_context_examples) > 0:
            more_guidance.append('To help your judgment, some examples are provided in [Examples].')
            in_context_prompt = ["[Examples]", "'''"]
            in_context_prompt.append(json.dumps(in_context_examples, ensure_ascii=False, indent=4))
            in_context_prompt.append("'''")
            in_context_prompt = '\n'.join(in_context_prompt)
        else:
            in_context_prompt = ""

        # 是否有更多需要补充的指南
        tmp = []
        for idx, guidance in enumerate(more_guidance):
            tmp.append('%s. %s' % (idx + 4, guidance))
        more_guidance = '\n'.join(tmp)

        # 根据评价指标设置评价的标准
        input_format = {}
        criteria = []
        for idx, metric in enumerate(metrics):
            tmp = EN_TEXT_EVAL_METRICS[metric]

            input_format[background] = '{{Background_VALUE}}'
            input_format[question] = '{{Question_VALUE}}'
            input_format[options] = '{{Options_VALUE}}'
            tmp = tmp.replace("{{Background}}", background)
            tmp = tmp.replace("{{Question}}", question)
            tmp = tmp.replace("{{Options}}", options)
            criteria.append('%d. %s' % (idx + 1, tmp))

        criteria = '\n'.join(criteria)

        # 给定输入的模板格式
        input_format = json.dumps(input_format, ensure_ascii=False, indent=4)

        self.meta_dict = {
            "{{DATATYPE}}": "str",
            "{{Background}}": background,
            "{{Question}}": question,
            "{{Options}}": options,
            "{{Language}}": language,
            "{{Output}}": output_pattern,
            "{{Input}}": input_format,
            "{{Criteria}}": criteria,
            "{{TASK_NAME}}": task_name,
            "{{MORE_GUIDANCE}}": more_guidance,
            "{{MORE_TASK_DEFINITION}}": more_task_definition,
            "{{In-Context Examples}}": in_context_prompt
        }

    def judge_a_case(self, case_data):
        llm_model = self.llm_model
        repeat_times = -1

        while True:
            repeat_times += 1
            if repeat_times >= llm_model.max_retries:
                break
            # 首先构造prompt
            prompt = llm_model.fit_case(pattern=self.prompt_pattern, data=case_data, meta_dict=self.meta_dict)
            contexts = llm_model.create_prompt(prompt)
            results = llm_model.request_llm(contexts, repeat_times=repeat_times)

            if results is not None and results[-1]['role'] == 'assistant':
                extracted_results = self.extract_scores(results[-1]['content'])
                if extracted_results is not None:
                    return prompt, extracted_results

        return None

    def judge_cases(self, cases_data):
        results = []
        for case_data in cases_data:
            prompt, result = self.judge_a_case(case_data)
            if result:
                results.append(result)
        return results

    # def extract_scores(self, last_response: str):
    #     try:
    #         last_response = last_response.strip('\r\n')
    #         # 找到JSON字符串的开始和结束位置
    #         start = last_response.find('{')
    #         end = last_response.rfind('}') + 1
    #
    #         # 提取JSON字符串
    #         json_str = last_response[start:end]
    #         json_str = json_str.replace("\n", "")
    #
    #         # 将JSON字符串解析为Python字典
    #         # print(json_str)
    #         data_dict = json.loads(json_str)
    #
    #         res_dict = {}
    #
    #         # 随后需要根据Output的Pattern做检查
    #         for k, v in self.result_pattern.items():
    #             # 首先检查是否存在，不存在return None
    #             if k not in data_dict:
    #                 return None
    #             # 检查值的有效性
    #
    #             value = data_dict[k]
    #             if isinstance(value, str):
    #                 value = value.strip('\r\n')
    #
    #             strict_flag = False
    #             data_type = str
    #             items = v.split("_")
    #
    #             # 判定是否严格模式
    #             if items[0].upper() == items[0]:
    #                 strict_flag = True
    #
    #             # 检查完成
    #             res_dict[k] = value
    #
    #         return res_dict
    #     except Exception as e:
    #         raise e
    #         return None

    def extract_scores(self, last_response: str):
        try:
            results = []
            responses = last_response.strip('\r\n').split('\n\n')
            for response in responses:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end].replace("\n", "")
                data_dict = json.loads(json_str)

                res_dict = {}
                for k, v in self.result_pattern.items():
                    if k not in data_dict:
                        return None

                    value = data_dict[k].strip('\r\n') if isinstance(data_dict[k], str) else data_dict[k]
                    res_dict[k] = value
                results.append(res_dict)
            return results
        except Exception as e:
            raise e
            return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../remote/configs/glm3.json")
    args = parser.parse_args()

    # 定义一个Logger
    logger = set_logger("glm_tmp.log")

    # 定义一个Agent
    glm3 = ChatGLMLLM(args.config_path)

    # 定义参数
    task_name = "Choose Answer Based On Context and Question"
    background = "Background"
    question = "Question"
    options = "Options"
    language = "Chinese"
    # language = "English"

    result_pattern = {
        "Answer": "str_",
        "Analysis": "str_"
    }

    more_guidance = ['selecting the answer depends on context,not guesswork.']

    in_context_examples = [
        {
            "Input": {
                "Background": "某商店进行了一次促销活动，对购买商品的顾客进行了抽奖。抽奖规则是：购买金额每满100元，就可以获得一次抽奖机会。小红购买了一件价值280元的商品。",
                "Question": " 小红在这次抽奖活动中获得了几次抽奖机会？",
                "Options":"A. 2 B. 3 C. 4 D. 5",
            },
            "Output": {
                "Answer": "A. 2",
                "Analysis": "小红购买的商品价值为280元，因此可以获得2次抽奖机会（每满100元获得1次）。所以小红在这次抽奖活动中获得了2次抽奖机会。"
            }
        }
    ]

    judge_agent = JudgeAgent(logger, glm3, task_name, result_pattern,
                               language=language, background=background, question=question,options=options,
                               more_guidance=more_guidance, in_context_examples=in_context_examples)
    multiple_data = [
        {
            "{{Background_VALUE}}": "下面有四个数字",
            "{{Question_VALUE}}": "哪个数是一个质数？",
            "{{Options_VALUE}}": "A.12  "
                                 "B.17 "
                                 "C.22 "
                                 "D.28"
        },

        {
            "{{Background_VALUE}}": "马克思指出,判断一个变革时代不能以该时代的意识为依据,相反,这个意识必须从物质生活的矛盾中去解释。",
            "{{Question_VALUE}}": "这里的“物质生活的矛盾”从根本上说是",
            "{{Options_VALUE}}": "A.社会生产力和生产关系的现存冲突  "
                                 "B.经济基础与上层建筑的现存冲突 "
                                 "C.人类社会与自然界的现存冲突 "
                                 "D.社会存在与社会意识的现存冲突"
        },

        {
            "{{Background_VALUE}}": "马克思、恩格斯始终站在革命斗争的最前沿，他们的一生是为推翻旧世界，建立新世界而不息战斗的一生。",
            "{{Question_VALUE}}": "马克思恩格斯领导创建的世界上第一个无产阶级政党是?",
            "{{Options_VALUE}}": "A.国际工人协会 "
                                 "B.正义者联盟 "
                                 "C.共产主义者同盟 "
                                 "D.社会主义工人国际"
        },

        {
            "{{Background_VALUE}}": "“橘生淮南则为橘，生为淮北则为枳，叶徒相似，其实味不同。所以然者何?水土异也。”",
            "{{Question_VALUE}}": "橘逾淮为枳说明了?",
            "{{Options_VALUE}}": "A.事物的发展变化以时间地点和条件为转移 "
                                 "B.事物的普遍联系是通过中介来实现的 "
                                 "C.任何实体事物都是普遍联系之网上的一个网结 "
                                 "D.事物的变化和发展是一个过程"
        },
    ]
    # prompt, res = judge_agent.judge_a_case(data)
    # print("本次请求的Prompt是", prompt)
    # print("本次请求的结果是", res)

    results = judge_agent.judge_cases(multiple_data)
    for idx, result in enumerate(results):
        print(f"第{idx + 1}个案例的结果：{result}")
