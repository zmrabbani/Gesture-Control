from datetime import datetime
import locale
locale.setlocale(locale.LC_TIME, "id_ID") # swedish

def get_part_of_day(h):
    if 5 <= h <= 11:
        return "Pagi"
    elif 12 <= h <= 16 :
        return "Siang"
    elif 16 <= h <= 18 :
        return "Sore"
    else:
        return "Malam"

def get_time(time):
    waktu = datetime.now()
    waktu = waktu.strftime(f"{time}")
    return waktu

part = get_part_of_day(datetime.now().hour)
