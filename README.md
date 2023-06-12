# Final-Project-ML-middle
# Распознавание эмоций на фото, видео и с web-камеры

<p align="center"><img src="/imgs/Эмоции.webp" width="500" alt="Эмоции"></p>

## Пошаговая инструкция по установке и запуску модели распознавания эмоций
  * [Шаг 1. Создание виртуальной среды](#создание_виртуальной_среды)
  * [Шаг 2. Подходы к решению задачи](#подходы-к-решению-задачи)
## Создание виртуальной среды
Сразу оговоримся, что модель создавалась в операционной системе UBUNTU 20.04.6 LTS. Для другой операционной системы потребуются корректировки каких то команд.\
Итак, создадим папку под проект. Назовите ее как вам будет удобно. Клонируйте этот репозиторий в созданную папку: https://github.com/mic21053/Final-Project-ML-middle  
Обратите внимание - так как github не дает загружать большие по объему файлы, в папках train, test_kaggle, My_model и MyVA_model находится не содержимое этих папок, а текстовые файлы со ссылками, по которым можно скачать соответствующий контент и поместить его в эти папки.  
Далее создаем виртуальную среду:
<pre>
python -m venv <название вашей виртуальной среды>
</pre>
Активируем её:
<pre>
source <название вашей виртуальной среды>/bin/activate
</pre>
Создаем зависимости и добавляем виртуальную среду в ядро jupyter notebook:
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=<название вашей виртуальной среды>
</pre>

## Подходы к решению задачи
### Задача классификации


