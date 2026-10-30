import numpy as np

def distillation_loss(
	student_logits: np.ndarray,
	teacher_logits: np.ndarray,
	temperature: float = 1.0
) -> float:
	"""
	Compute knowledge distillation loss.
	
	L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
	
	Args:
		student_logits: Logits from student model (shape: [num_classes] or [batch_size, num_classes])
		teacher_logits: Logits from teacher model (shape matches student_logits)
		temperature: Softmax temperature
		
	Returns:
		Distillation loss value (scalar float)
	"""
	# 1. Scale logits by the temperature factor
	s_scaled = student_logits / temperature
	t_scaled = teacher_logits / temperature
	
	# 2. Compute Stable Softmax for the Teacher
	t_max = np.max(t_scaled, axis=-1, keepdims=True)
	t_exp = np.exp(t_scaled - t_max)
	p_teacher = t_exp / np.sum(t_exp, axis=-1, keepdims=True)
	
	# 3. Compute Stable Softmax for the Student
	s_max = np.max(s_scaled, axis=-1, keepdims=True)
	s_exp = np.exp(s_scaled - s_max)
	q_student = s_exp / np.sum(s_exp, axis=-1, keepdims=True)
	
	# 4. Handle numerical stability for log calculations (prevent log(0))
	eps = 1e-12
	p_teacher = np.clip(p_teacher, eps, 1.0)
	q_student = np.clip(q_student, eps, 1.0)
	
	# 5. Compute Kullback-Leibler (KL) Divergence 
	# KL(P || Q) = sum(P * log(P / Q))
	# Note: Axis -1 isolates computation to the distribution per sample
	kl_div = np.sum(p_teacher * np.log(p_teacher / q_student), axis=-1)
	
	# 6. Average across the batch if batch input is given, and multiply by T^2
	mean_kl = np.mean(kl_div)
	loss = (temperature ** 2) * mean_kl
	
	return float(loss)