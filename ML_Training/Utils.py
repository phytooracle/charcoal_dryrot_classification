from tensorflow.keras import backend as K

def weighted_dice_coef(y_true, y_pred):

	w_1 = 185
	w_0 = 1

	y_true_f_1 = K.flatten(y_true)
	y_pred_f_1 = K.flatten(y_pred)
	y_true_f_0 = K.flatten(1-y_true)
	y_pred_f_0 = K.flatten(1-y_pred)

	intersection_0 = K.sum(y_true_f_0 * y_pred_f_0)
	intersection_1 = K.sum(y_true_f_1 * y_pred_f_1)

	return 1-2 * (w_0 * intersection_0 + w_1 * intersection_1) / ((w_0 * (K.sum(y_true_f_0) + K.sum(y_pred_f_0))) + (w_1 * (K.sum(y_true_f_1) + K.sum(y_pred_f_1))))
