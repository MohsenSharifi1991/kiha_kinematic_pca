import matplotlib;

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'


# E:\\Media\\ffmpeg\\bin\\ffmpeg.exe

fig, ax = plt.subplots()

x = np.arange(0, 2 * np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i / 10.0))  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,


ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=1, blit=True)

mywriter = animation.FFMpegWriter()
ani.save('mymovie.mp4', writer=mywriter)
# plt.show()
#########################################################################
##########################################################################
# import numpy as np
# import pandas as pd
# url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
# df = pd.read_csv(url, delimiter=',', header='infer')
# df_interest = df.loc[
#     df['Country/Region'].isin(['United Kingdom', 'US', 'Italy', 'Germany'])
#     & df['Province/State'].isna()]
# df_interest.rename(
#     index=lambda x: df_interest.at[x, 'Country/Region'], inplace=True)
# df1 = df_interest.transpose()
# df1 = df1.drop(['Province/State', 'Country/Region', 'Lat', 'Long'])
# df1 = df1.loc[(df1 != 0).any(1)]
# df1.index = pd.to_datetime(df1.index)
# fig,ax = plt.subplots()
# explode=[0.01,0.01,0.01,0.01] #pop out each slice from the pie
# def getmepie(i):
#     def absolute_value(val): #turn % back to a number
#         a  = np.round(val/100.*df1.head(i).max().sum(), 0)
#         return int(a)
#     ax.clear()
#     plot = df1.head(i).max().plot.pie(y=df1.columns,autopct=absolute_value, label='',explode = explode, shadow = True)
#     plot.set_title('Total Number of Deaths\n' + str(df1.index[min( i, len(df1.index)-1 )].strftime('%y-%m-%d')), fontsize=12)
#
# animator = animation.FuncAnimation(fig, getmepie, interval = 1)
# video = anim.to_html5_video()
# FFwriter = animation.FFMpegWriter()
# animator.save('animation.mp4', writer = FFwriter, fps=10)
# # animator.save('myfirstAnimation.gif')
# # plt.show()
#
# animator.save('myfirstAnimation.gif')
