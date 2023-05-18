import logging
import os

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from PIL import Image
from telegram.ext import CommandHandler, Filters, MessageHandler, Updater

from birds_name import bird_name

load_dotenv()
secret_token = os.getenv("TOKEN")

updater = Updater(token=secret_token)


def wake_up(update, context):
    chat = update.effective_chat
    name = update.message.chat.first_name
    context.bot.send_message(
        chat_id=chat.id,
        text=(
            "Привет {}! Присылай фото птицы и бот расскажет, чем её кормить."
            " ВНИМАНИЕ! Для корректной работы бота необходимо отправлять"
            " максимально приближенное фото птицы(чем меньше фона - тем лучше)"
        ).format(
                name
         ),
    )

def bird_predict(update, context):
    model = load_bmodel()
    user = update.message.from_user
    photo_file = update.message.photo[-1].get_file()
    photo_file.download("user_photo.jpg")
    update.message.reply_text("Cекундочку, бот обрабатывает изображение...")
    update.message.reply_text(get_prediction(model, "user_photo.jpg"))

def load_bmodel():
    model = load_model("birds_bot/my_birds")
    return model

def preprocess_image(image):
    img_datagen = ImageDataGenerator(rescale=1.0 / 255)
    height = 224
    width = 224
    image_resized = tf.image.resize_with_pad(
        image, target_height=height, target_width=width
    )
    x = img_to_array(image_resized)
    x = x.reshape((1,) + x.shape)
    img_generator = img_datagen.flow(x, batch_size=1)

    return img_generator

def get_prediction(model, image_filename):
    image = Image.open(image_filename)
    image_preprocessed = preprocess_image(image)
    preds = model.predict(image_preprocessed)
    pred = np.argmax(preds)
    try:
        return bird_name(pred.tolist())
    except KeyError:
        updater.dispatcher.add_error_handler(error_program)

def error_program(update, context):
    chat = update.effective_chat
    message = (
        "Упс! Бот не смог определить птичку :(."
        "Пожалуйста, отправьте новое фото."
    )
    context.bot.send_message(chat_id=chat.id, text=message)

def main():
    while True:
        try:
            updater.dispatcher.add_handler(CommandHandler("start", wake_up))
            updater.dispatcher.add_handler(
                MessageHandler(Filters.photo, bird_predict)
            )
        except Exception as error:
            logging.error(f"Сбой в работе программы: {error}")
            updater.dispatcher.add_error_handler(error_program)
        finally:

            updater.start_polling(poll_interval=15.0)
            updater.idle()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    main()
