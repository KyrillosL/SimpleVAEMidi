import vae_evaluator


vae = vae_evaluator.Vae()
data = vae.load_all_data("/home/kyrillos/CODE/VAEMIDI/quantized_rythm_dataset_v2_temperature/100" , 1, 0)


data += vae.load_all_data("/home/kyrillos/CODE/VAEMIDI/quantized_rythm_dataset_v2_temperature/0" , 0, 0)


vae.get_coord(data, 128,False)
#print(vae.get_distance( "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/to_compare_0/random_1.mid" ))