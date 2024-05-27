
import json
import logging
from applications.prompts import EN_TEXT_EVAL_GENERAL_PROMPT_PATTERN, EN_TEXT_EVAL_METRICS
from remote import RemoteLLMs



class JudgeAgent:
    def __init__(self, logger, llm_model: RemoteLLMs, task_name: str,metrics: dict,
                 language,  question,  in_context_examples=[],
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


            input_format[question] = '{{Question_VALUE}}'


            tmp = tmp.replace("{{Question}}", question)

            criteria.append('%d. %s' % (idx + 1, tmp))

        criteria = '\n'.join(criteria)

        # 给定输入的模板格式
        input_format = json.dumps(input_format, ensure_ascii=False, indent=4)

        self.meta_dict = {
            "{{DATATYPE}}": "str",

            "{{Question}}": question,

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
                self.logger.error('LLM Model repeat times exceed the limit.')
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