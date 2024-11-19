# Локальное запуск проекта


## 1. Установка onnxruntime.

### Установка на ArchLinux
``` zsh
sudo pacman -Suy
sudo pacman -S onnxruntime
```

---

## 2. Скачивание моделей.

Необходимый архив с моделями можно установить по [ссылке](https://disk.yandex.ru/d/10OegiujsUYQ1A).\
Расположение моделей на копьютере не имеет значения.

---

## 3. Описание переменных окружения.

В корневой директории проекта необходимо создать файл **config.toml** и указать следующею информацию:

``` toml
[service]
host = "0.0.0.0"
port = 3003
swagger_path = "/swagger-ui" # пусть к докумантации свагер после запуска проекта
body_limit = 100000000 # максимальный размер загружаемых файлов на сервер (в байтах)


[model.facial_processing.detector]
model_path = "{путь к директории 'models'}/models/antelopev2/detection/model.onnx"
model_name = "detector"


[model.facial_processing.recognizer]
model_path = "{путь к директории 'models'}/models/antelopev2/recognition/model.onnx"
model_name = "recognizer"


[model.search.textual]
model_path = "{путь к директории 'models'}/models/clip/text/model.onnx"
model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


[model.search.visual]
model_path = "{путь к директории 'models'}/models/clip/image/model.onnx"
model_name = "visual"
```

---

## 4. Сборка проекта.

``` zsh
cargo build
```

---

## 5. Запуск проекта.

``` zsh
cargo run
```

---
---

# Тестирование проекта


## 1. Запуск тестов.

``` zsh
cargo test
```

---
---

# Использование

Для просмотра и использования реализованной логики необходимо перейти по ссылку, \
которая будет составлена на основании указанных вами переменных окружения в файле **config.toml**.

Так например при использовании настроек по умолчанию вам необходимо перейти по ссылке **http://0.0.0.0:3003/swagger-ui**
