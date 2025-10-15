	# streamlit_app.py
	import streamlit as st
	import requests
	import json
		# --- 1. 提示词模板 (定义GLM-4.6的角色和任务) ---
	SYSTEM_PROMPT = """
	你是一位顶尖的美国移民律师顾问，同时精通学术数据分析和USCIS（美国公民及移民服务局）的NIW（国家利益豁免）和EB-1A（杰出人才）申请审查逻辑。
	你的任务是分析用户提供的科研背景资料，并模拟USCIS的审查标准，给出一份全面、专业的评估报告。
	你必须严格遵循以下输出格式要求：
	1.  你的全部回答必须是一个单一、有效的JSON对象。
	2.  不要在JSON对象前后添加任何解释性文字、代码块标记（如```json）或任何其他内容。
	3.  如果无法分析，请返回一个包含 "error": "Unable to process the input." 的JSON对象。
	JSON对象的结构必须如下：
	{
	  "analysis_summary": {
	    "field_of_expertise": "申请人的专业领域",
	    "key_achievements": "主要成就的简要总结"
	  },
	  "prong_analysis": {
	    "prong_1": {
	      "score": 8,
	      "reasoning": "论证过程，解释为什么满足或不满足Substantial Merit and National Importance。",
	      "suggestions": "针对Prong 1的补强建议。"
	    },
	    "prong_2": {
	      "score": 7,
	      "reasoning": "论证过程，解释为什么申请人Well-Positioned to Advance the Field。",
	      "suggestions": "针对Prong 2的补强建议。"
	    },
	    "prong_3": {
	      "score": 9,
	      "reasoning": "论证过程，解释为什么Benefit to the U.S. outweighs Labor Certification。",
	      "suggestions": "针对Prong 3的补强建议。"
	    }
	  },
	  "overall_assessment": {
	    "total_score": 24,
	    "success_probability_niw": "高",
	    "success_probability_eb1a": "中",
	    "overall_suggestions": "综合补强建议，包括证据类型、寻找推荐信的人选等。"
	  },
	  "future_plan_draft": [
	    "第一条未来计划，与申请人研究方向衔接，体现美国国家意义。",
	    "第二条未来计划，语言积极、可信、有行动路线。",
	    "第三条未来计划，可以涉及教学、合作或技术转化。"
	  ]
	}
	"""
	USER_PROMPT_TEMPLATE = """
	请根据以下申请人提供的科研资料，进行NIW/EB-1A智能评估。
	申请人科研资料：
	---
	{user_input}
	---
	请严格按照系统提示词中定义的JSON格式输出你的评估报告。
	"""
	# --- 2. 核心函数 (用于调用GLM-4.6 API) ---
	API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
	# 在部署时，我们会通过Streamlit Secrets来管理这个密钥
	# 在本地测试时，你可以直接在这里填入你的API Key
	API_KEY = st.secrets["ZHIPU_API_KEY"] 
	def get_glm_assessment(user_input_text: str) -> dict:
	    """
	    调用GLM-4.6模型获取NIW/EB-1A评估报告
	    Args:
	        user_input_text (str): 用户输入的科研资料
	    Returns:
	        dict: 解析后的JSON评估报告，如果失败则返回错误信息
	    """
	    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
	    user_prompt = USER_PROMPT_TEMPLATE.format(user_input=user_input_text)
	    payload = {"model": "glm-4.6", "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}], "temperature": 0.2}
	    try:
	        response = requests.post(API_URL, headers=headers, json=payload)
	        response.raise_for_status()  # 如果请求失败则抛出异常
	        content = response.json()['choices'][0]['message']['content']
	        assessment_report = json.loads(content)
	        return assessment_report
	    except requests.exceptions.RequestException as e:
	        return {"error": f"API request failed: {e}"}
	    except json.JSONDecodeError:
	        return {"error": "Failed to decode JSON from model response.", "raw_content": content if 'content' in locals() else "N/A"}
	    except KeyError:
	        return {"error": "Unexpected API response structure.", "raw_response": response.text}
	# --- 3. Streamlit 前端界面 ---
	st.set_page_config(page_title="NIW/EB-1A 智能评估助手", layout="wide")
	st.title("🧑‍⚖️ NIW / EB-1A 智能评估助手")
	st.markdown("基于 GLM-4.6 模型，模拟 USCIS 审查逻辑，为您的申请提供专业评估和建议。")
	# 使用 st.session_state 来缓存用户输入，避免页面刷新时丢失
	if 'user_input' not in st.session_state:
	    st.session_state.user_input = ""
	with st.form("assessment_form"):
	    st.subheader("第一步：请输入您的科研资料")
	    user_input = st.text_area(
	        "请详细描述您的研究方向、发表的重要论文（期刊、影响因子、引用次数）、审稿经历、获奖情况、国际合作等。信息越详细，评估越准确。",
	        height=300,
	        value=st.session_state.user_input
	    )
	    submitted = st.form_submit_button("开始智能评估")
	    if submitted:
	        if not user_input.strip():
	            st.warning("请输入您的科研资料后再进行评估。")
	        else:
	            st.session_state.user_input = user_input
	            with st.spinner('🤖 GLM-4.6 正在分析中，请稍候...'):
	                report = get_glm_assessment(user_input)
	            if "error" in report:
	                st.error(f"分析过程中出现错误: {report['error']}")
	                if "raw_content" in report:
	                    with st.expander("查看模型原始输出"):
	                        st.code(report['raw_content'])
	            else:
	                st.success("✅ 评估完成！")
	                # --- 4. 结果展示 ---
	                st.header("评估报告")
	                # 分析摘要
	                st.subheader("📄 分析摘要")
	                col1, col2 = st.columns(2)
	                col1.metric("专业领域", report['analysis_summary']['field_of_expertise'])
	                col2.metric("关键成就", report['analysis_summary']['key_achievements'])
	                # Prong 分析
	                st.subheader("⚖️ USCIS Prong 详细分析")
	                for prong_name, prong_data in report['prong_analysis'].items():
	                    with st.expander(f"Prong {prong_name.split('_')[-1]}: Score {prong_data['score']}/10"):
	                        st.write(f"**论证逻辑:** {prong_data['reasoning']}")
	                        st.info(f"**补强建议:** {prong_data['suggestions']}")
	                # 综合评估
	                st.subheader("📊 综合评估")
	                col_niw, col_eb1a, col_score = st.columns(3)
	                col_niw.metric("NIW 成功概率", report['overall_assessment']['success_probability_niw'])
	                col_eb1a.metric("EB-1A 成功概率", report['overall_assessment']['success_probability_eb1a'])
	                col_score.metric("综合得分", report['overall_assessment']['total_score'])
	                st.write(f"**综合建议:** {report['overall_assessment']['overall_suggestions']}")
	                # Future Plan
	                st.subheader("🧭 Future Plan 写作建议")
	                for i, plan in enumerate(report['future_plan_draft']):

	                    st.write(f"{i+1}. {plan}")
