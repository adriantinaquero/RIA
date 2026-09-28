class PIDController():
    """Esta clase se usará en ambos ejercicios para poder manejar más fácilmente varos
    controladores a la vez."""
    def __init__(self, kp:float, ki:float, kd:float, errorFunc, correctionFunc, robobo):
        self.kp = kp # Multiplicador del error proporcional
        self.ki = ki # Multiplicador de la integral
        self.kd = kd # Multiplicador de la derivada
        self.errorFunc = errorFunc # Función de cálculo de error
        self.correctionFunc = correctionFunc # Función de corrección del error
        self.rob = robobo # Robobo de referencia
        self.error_previo = 0 
        self.integral = 0

    def update(self):
        error = self.errorFunc(self.rob)
        self.integral += error
        derivada = error - self.error_previo
        correcion = round(error * self.kp 
                          + self.integral * self.ki 
                          + derivada * self.kd)
        self.correctionFunc(self.rob, correcion)
        self.error_previo = error
        print(error)
