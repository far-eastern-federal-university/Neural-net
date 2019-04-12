# Очевидно, что данный код работать не будет, если попытаться запустить его в таком виде

class NeuralNetwork():
    def __init__(self):
        #Запустить генератор случайных чисел, чтобы он генерировал те же числа каждый раз, когда программа запускается.
        random.seed(1)

        #Мы моделируем один нейрон с 3 входными и 1 выходным соединением
        # Мы присваиваем случайные веса матрице 3 x 1 со значениями в диапазоне от -1 до 1
        #and mean 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    #Сигмовидная функция, которая описывает S-образную кривую
    # Мы передаем взвешенную сумму входов через эту функцию
    #нормализуем их от 0 до 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    #Производная сигмоидной функции
    #Это градиент сигмовидной кривой
    #Это показывает, насколько мы уверены в существующем весе
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    #Мы обучаем нейронную сеть в процессе проб и ошибок
    #Регулировка синаптических весов каждый раз
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            #Пройдите обучение через нашу нейронную сеть (один нейрон)
            output = self.think(training_set_inputs)

            #Рассчитать ошибку (Разница между желаемым выводом
            #и прогнозируемый результат)
            error = training_set_outputs - output

            #Умножим ошибку на вход и снова на градиент сигмоидной кривой
            #Это означает, что менее уверенные веса корректируются больше
            #Это означает, что входные данные, которые равны нулю, не вызывают изменения весов
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #Отрегулируем вес
            self.synaptic_weights += adjustment

    #Нейронная сеть думает
    def think(self, inputs):
        #Передача входных данных через нашу нейронную сеть (наш единственный нейрон)
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Запуск одной нейронной сети нейронов
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ", neural_network.synaptic_weights)

    #Учебный комплект. У нас есть 4 примера, каждый из которых состоит из 3 входных значений
    #и 1 выходное значение
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    #Тренируем нейронную сеть, используя тренировочный набор
    #Сделаем это 10000 раз и вносим небольшие корректировки каждый раз
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("Новые синаптические веса после тренировки: ", neural_network.synaptic_weights)

    #Тестирование нейронной сети в новой ситуацииНовые синаптические веса после тренировки:
    print ( "Учитывая новую ситуацию [1, 0, 0] -> ?: ", neural_network.think(array([1, 0, 0])))
