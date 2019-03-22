import model
from tensorflow.keras.backend import clear_session

cfgs = [
    ('fc_rgb', ['rgb']),
    ('fc_rgb_laws', ['rgb', 'laws']),
    ('fc_rgb_haralick8', ['rgb', 'haralick8']),
    ('fc_rgb_haralick16', ['rgb', 'haralick16']),
    ('fc_rgb_haralick8+16', ['rgb', 'haralick8', 'haralick16']),
    ('fc_rgb_edge', ['rgb', 'edge']),
    ('fc_rgb_laws_edge', ['rgb', 'laws', 'edge']),
    ('fc_rgb_haralick8_edge', ['rgb', 'haralick8', 'edge']),
    ('fc_rgb_haralick16_edge', ['rgb', 'haralick16', 'edge']),
    ('fc_rgb_haralick8+16_edge', ['rgb', 'haralick8', 'haralick16', 'edge'])
]

def main():
    for name, features in cfgs:
        mod = model.Model(name, features)
        mod.save_model()
        mod.train()
        mod.save_weights()
        mod.save_history()
        mod.test()
        mod.save_stats()
        del mod
        clear_session()


if __name__ == '__main__':
    main()
