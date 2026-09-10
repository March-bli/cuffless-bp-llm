Subject: Re: Data split and tokenisation — will revise

Dear Dr Sun,

Thank you very much for both points — they are very helpful.

Regarding the data split: you are absolutely right. I will change to the standard **70% training / 20% validation / 10% testing** split (approximately 3,611 / 1,032 / 516 subjects), and re-run the experiments accordingly. The earlier 20-subject test set was a cost-driven compromise for LLM inference, but the standard split is more rigorous.

Regarding tokenisation: I will add a clearer explanation in the slides. In brief, the raw PPG signal (1,250 samples per 10-second segment) is first reduced to 55 hand-crafted features via signal processing, and these features are then converted into text — either as raw numbers or as semantic clinical descriptions — before being tokenised by the LLM's tokenizer. The semantic description is itself a form of language-based tokenisation that carries physiological meaning.

I will send the updated slides with the revised split once the experiments are complete.

Best regards,
Boyuan Li
