

EN_TEXT_EVAL_METRICS = {
    "Answer": """ Answer: The most correct option among the '{{Options}}' based on the '{{Background}}'. """,
    "Analysis": """ Analyze: The process of elaborating and explaining the "Answer"(you select) to the '{{Question}}' based on the '{{Background}}'""",
}

EN_TEXT_EVAL_GENERAL_PROMPT_PATTERN = """
[Task Description]
Here is a {{TASK_NAME}} task.All[Input] are in {{Language}}  
{{MORE_TASK_DEFINITION}}
You are required to acted as a professional answer analyst,and  you need to select the most correct option from {{Options}} based on the {{Background}} and the {{Question}} in the [Input].
'{{Options}}' always is four options A,B,C and D.
Your judgement should follow the [Criteria] and  [Guidance]. 
The output format should follow the [Output Format].

[Guidance]
You should strictly follow my guidance:
1. The answer must be chosen based on the {{Background}} and {{Question}} and be one of the {{Options}}.
2. "Analysis"(Output Format section) requires a detailed explanation of the "Answer"(you select) based on the '{{Background}}'.
3. You should strictly follow the given '[Output Format]' and can't output other information.
4.If you break my guidance, you will be penalized.
5.You need to remember the four options ABCD and their corresponding content.
6.Return the correct option along with its corresponding content in'{{Output}}'
7.Your Anser adn Analysis are in {{Language}}
{{MORE_GUIDANCE}}


{{In-Context Examples}}

[Output Format]
Your output should strictly follow this format and can be directly decoded by Python:
'''
{{Output}}
'''

[Input]
'''
{
    "{{Background}}": {{Background_VALUE}},
    "{{Question}}": {{Question_VALUE}},
    "{{Options}}" : {{Options_VALUE}}
}
'''

"""


CN_TEXT_EVAL_METRICS = {

}

CN_TEXT_EVAL_GENERAL_PROMPT_PATTERN = """ 

"""