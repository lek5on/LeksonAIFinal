using MO_31_2_Savchenko_LeksonAI.NeuroNet;
using System;
using System.Configuration;
using System.IO;
using System.Windows.Forms;


namespace MO_31_2_Savchenko_LeksonAI.NeuroNet
{
    abstract class Layer
    {
        //Поля
        protected string name_Layer; //название слоя
        string pathDirWeights; //путь к каталогу, где находится файл синаптических весов
        string pathFileWeights; //путь к файлу саниптическов весов
        protected int numofneurons; //число нейронов текущего слоя
        protected int numofprevneurons; //число нейронов предыдущего слоя
        protected const double learningrate = 0.2d; //скорость обучения
        protected const double momentum = 0.02d; //момент инерции
        protected double[,] lastdeltaweights; //веса предыдущей итерации
        protected Neuron[] neurons; //массив нейронов текущего слоя

        //Свойства
        public Neuron[] Neurons { get => neurons; set => neurons = value; }
        public double[] Data //Передача входных сигналов на нейроны слоя и авктиватор
        {
            set
            {
                for (int i = 0; i < numofneurons; i++)
                {
                    Neurons[i].Activator(value);
                }
            }
        }

        //Конструктор
        protected Layer(int non, int nopn, NeuronType nt, string nm_Layer)
        {
            numofneurons = non; //количество нейронов текущего слоя
            numofprevneurons = nopn; //количество нейронов предыдущего слоя
            Neurons = new Neuron[non]; //определение массива нейронов
            name_Layer = nm_Layer; //наиминование слоя
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";

            double[,] Weights; //временный массив синаптических весов
            lastdeltaweights = new double[non, nopn + 1];

            if (File.Exists(pathFileWeights)) //определяет существует ли pathFileWeights
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights); //считывает данные из файла
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }

            for (int i = 0; i < non; i++) //цикл формирования нейронов слоя и заполнения
            {
                double[] tmp_weights = new double[nopn + 1];
                for (int j = 0; j < nopn; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(tmp_weights, nt); //заполнение массива нейронами
            }
        }

        //Метод работы с массивом синаптических весов слоя
        public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';', ' ' };
            string[] tmpStrWeights;
            double[,] weights = new double[numofneurons, numofprevneurons + 1];

            switch (mm)
            {
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path);
                    string[] memory_elemnt;

                    for (int i = 0; i < numofneurons; i++)
                    {
                        memory_elemnt = tmpStrWeights[i].Split(delim);
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_elemnt[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                case MemoryMode.SET:
                    tmpStrWeights = new string[numofneurons];
                    for (int i = 0; i < numofneurons; i++)
                    {
                        string[] memory_elemnt2 = new string[numofprevneurons + 1];
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            memory_elemnt2[j] = neurons[i].Weights[j]
                                .ToString(System.Globalization.CultureInfo.InvariantCulture)
                                .Replace('.', ',');
                        }
                        tmpStrWeights[i] = string.Join(";", memory_elemnt2);
                    }
                    File.WriteAllLines(path, tmpStrWeights);
                    for (int i = 0; i < numofneurons; i++)
                    {
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = neurons[i].Weights[j];
                        }
                    }
                    break;

                case MemoryMode.INIT:
                    Random random = new Random();

                    for (int i = 0; i < numofneurons; i++)
                    {
                        double[] neuronWeights = new double[numofprevneurons + 1];
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            neuronWeights[j] = (random.NextDouble() * 2.0) - 1.0;
                        }

                        double mean = Calc_Average(neuronWeights);
                        double variance = Calc_Dispers(neuronWeights);
                        double stdDev = Math.Sqrt(variance);

                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            if (stdDev > 1e-10)
                            {
                                neuronWeights[j] = (neuronWeights[j] - mean) / stdDev;
                            }
                            else
                            {
                                neuronWeights[j] = 0.0;
                            }
                        }

                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = neuronWeights[j];
                        }
                    }

                    // Сохраняем в файл
                    string[] lines = new string[numofneurons];
                    for (int i = 0; i < numofneurons; i++)
                    {
                        string[] row = new string[numofprevneurons + 1];
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            row[j] = weights[i, j]
                                .ToString(System.Globalization.CultureInfo.InvariantCulture)
                                .Replace('.', ',');
                        }
                        lines[i] = string.Join(";", row);
                    }
                    File.WriteAllLines(path, lines);
                    break;
            }

            return weights;
        }
        //метод расчета среднего
        protected double Calc_Average(double[] arr)
        {
            if (arr == null || arr.Length == 0)
                return 0;

            double sum = 0;
            for (int i = 0; i < arr.Length; i++)
            {
                sum += arr[i];
            }
            return sum / arr.Length;
        }

        //метод расчета дисперсии
        protected double Calc_Dispers(double[] arr)
        {
            if (arr == null || arr.Length == 0)
                return 0;

            double average = Calc_Average(arr);
            double sumSquares = 0;

            for (int i = 0; i < arr.Length; i++)
            {
                double deviation = arr[i] - average;
                sumSquares += deviation * deviation;
            }

            return sumSquares / arr.Length;
        }
        abstract public void Recognize(Network net, Layer nextLayer);
        abstract public double[] BackwardPass(double[] staff);
    }
    
}
