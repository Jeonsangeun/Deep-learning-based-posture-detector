import numpy as np

aa0 = np.load('./data_set/good_data1.npy')
aa1 = np.load('./data_set/normal_data1.npy')
aa2 = np.load('./data_set/bad_data1.npy')
aa3 = np.load('./data_set/bad_down_data1.npy')
aa4 = np.load('./data_set/right_up_data1.npy')
aa5 = np.load('./data_set/left_up_data1.npy')
aa6 = np.load('./data_set/right_down_data1.npy')
aa7 = np.load('./data_set/left_down_data1.npy')

bb0 = np.load('./data_set/good_data2.npy')
bb1 = np.load('./data_set/normal_data2.npy')
bb2 = np.load('./data_set/bad_data2.npy')
bb3 = np.load('./data_set/bad_down_data2.npy')
bb4 = np.load('./data_set/right_up_data2.npy')
bb5 = np.load('./data_set/left_up_data2.npy')
bb6 = np.load('./data_set/right_down_data2.npy')
bb7 = np.load('./data_set/left_down_data2.npy')

cc0 = np.load('./data_set/good_data3.npy')
cc1 = np.load('./data_set/normal_data3.npy')
cc2 = np.load('./data_set/bad_data3.npy')
cc3 = np.load('./data_set/bad_down_data3.npy')
cc4 = np.load('./data_set/right_up_data3.npy')
cc5 = np.load('./data_set/left_up_data3.npy')
cc6 = np.load('./data_set/right_down_data3.npy')
cc7 = np.load('./data_set/left_down_data3.npy')


dd0 = np.load('./data_set/good_data4.npy')
dd1 = np.load('./data_set/normal_data4.npy')
dd2 = np.load('./data_set/bad_data4.npy')
dd3 = np.load('./data_set/bad_down_data4.npy')
dd4 = np.load('./data_set/right_up_data4.npy')
dd5 = np.load('./data_set/left_up_data4.npy')
dd6 = np.load('./data_set/right_down_data4.npy')
dd7 = np.load('./data_set/left_down_data4.npy')


zz0 = np.load('./test_set/good_data.npy')
zz1 = np.load('./test_set/normal_data.npy')
zz2 = np.load('./test_set/bad_data.npy')
zz3 = np.load('./test_set/bad_down_data.npy')
zz4 = np.load('./test_set/right_up_data.npy')
zz5 = np.load('./test_set/left_up_data.npy')
zz6 = np.load('./test_set/right_down_data.npy')
zz7 = np.load('./test_set/left_down_data.npy')


# ff = np.load('save_data_set.npy')
# gg = np.load('save_data2_set.npy')

print(aa0.shape)
aa0 = np.reshape(aa0, [50, -1])
aa1 = np.reshape(aa1, [50, -1])
aa2 = np.reshape(aa2, [50, -1])
aa3 = np.reshape(aa3, [50, -1])
aa4 = np.reshape(aa4, [50, -1])
aa5 = np.reshape(aa5, [50, -1])
aa6 = np.reshape(aa6, [50, -1])
aa7 = np.reshape(aa7, [50, -1])
aa = np.array([aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7])

bb0 = np.reshape(bb0, [50, -1])
bb1 = np.reshape(bb1, [50, -1])
bb2 = np.reshape(bb2, [50, -1])
bb3 = np.reshape(bb3, [50, -1])
bb4 = np.reshape(bb4, [50, -1])
bb5 = np.reshape(bb5, [50, -1])
bb6 = np.reshape(bb6, [50, -1])
bb7 = np.reshape(bb7, [50, -1])
bb = np.array([bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7])

cc0 = np.reshape(cc0, [50, -1])
cc1 = np.reshape(cc1, [50, -1])
cc2 = np.reshape(cc2, [50, -1])
cc3 = np.reshape(cc3, [50, -1])
cc4 = np.reshape(cc4, [50, -1])
cc5 = np.reshape(cc5, [50, -1])
cc6 = np.reshape(cc6, [50, -1])
cc7 = np.reshape(cc7, [50, -1])
cc = np.array([cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7])

dd0 = np.reshape(dd0, [50, -1])
dd1 = np.reshape(dd1, [50, -1])
dd2 = np.reshape(dd2, [50, -1])
dd3 = np.reshape(dd3, [50, -1])
dd4 = np.reshape(dd4, [50, -1])
dd5 = np.reshape(dd5, [50, -1])
dd6 = np.reshape(dd6, [50, -1])
dd7 = np.reshape(dd7, [50, -1])
dd = np.array([dd0, dd1, dd2, dd3, dd4, dd5, dd6, dd7])

temp = []

zz0 = np.reshape(zz0, [10, -1])
zz1 = np.reshape(zz1, [10, -1])
zz2 = np.reshape(zz2, [10, -1])
zz3 = np.reshape(zz3, [10, -1])
zz4 = np.reshape(zz4, [10, -1])
zz5 = np.reshape(zz5, [10, -1])
zz6 = np.reshape(zz6, [10, -1])
zz7 = np.reshape(zz7, [10, -1])
zz = np.array([zz0, zz1, zz2, zz3, zz4, zz5, zz6, zz7])
print("test_size", zz.shape)
# np.save('test_data', zz)


print("train_1:", aa.shape)
# ff_dataframe = pd.DataFrame(ff)
# ff_dataframe.to_csv("dd.csv", header=False, index=False)
#
#
# gg_dataframe = pd.DataFrame(gg)
# gg_dataframe.to_csv("ee.csv", header=False, index=False)

for i in range(8):
    temp = np.hstack([aa, bb, cc, dd])

print("train_shape", temp.shape)

#print(test_data.shape)
# test_side = np.vstack([bb, bb1])
#
# test_waist = np.vstack([cc, cc1, cc2])


np.save('train_data2', temp)
# np.save('side_train_data', test_side)
# np.save('waist_train_data', test_waist)
#

temp2 = np.identity(8, dtype=float)
temp1 = np.array(range(8))

# np.save('posture1', temp1)

