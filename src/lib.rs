mod utils;
use js_sys::{Array as JSArray, Float32Array as JSFloat32Array};
use mel_spec::mel::mel;
use ndarray::{concatenate, s, Array1, Array2, Array3, Axis, Zip};
use ndarray_stats::QuantileExt;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::cmp::min;
use std::io::{BufReader, Cursor};
use wasm_bindgen::prelude::*;
use wavers::{IntoNdarray, ReadSeek, Wav};

// #[wasm_bindgen]
// extern "C" {
//     fn alert(s: &str);
//     #[wasm_bindgen(js_namespace = console, js_name = log)]
//     fn log(b: &[u8]);
//     #[wasm_bindgen(js_namespace = console, js_name = log)]
//     fn log_str(s: &str);
//     #[wasm_bindgen(js_namespace = console, js_name = log)]
//     fn log_jsval(s: &JsValue);
//     #[wasm_bindgen(js_namespace = console, js_name = log)]
//     fn log_jsvecf32(s: Vec<f32>);
//     #[wasm_bindgen(js_namespace = console, js_name = log)]
//     fn log_jsvecusize(s: Vec<usize>);
// }

// #[wasm_bindgen]
// pub fn greet() {
//     alert("Hello, indicasr-wasm!");
// }

fn do_premphasis(audio: Array2<f32>, val: f32) -> Array2<f32> {
    let arr1 = audio.column(0).insert_axis(Axis(1));
    let arr2 = audio.slice(s![.., 1..]).to_owned() - audio.slice(s![.., ..-1]).mapv(|a| a * val);
    let final_arr = concatenate(Axis(1), &[arr1, arr2.view()]).unwrap();
    final_arr
}

fn stft(
    signal: Array2<f32>,
    window_size: usize,
    step_size: usize,
    fft_size: usize,
) -> Array3<Complex32> {
    let num_channels = signal.dim().0;

    let signal_len = signal.len();
    let num_windows = (signal_len - window_size) / step_size + 1;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let hann_window: Array1<f32> = Array1::linspace(0., 1., window_size)
        .mapv(|x| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos());

    let mut stft_result = Array3::zeros((num_channels, num_windows, fft_size / 2 + 1));

    for c in 0..num_channels {
        for (i, window) in signal
            .row(c)
            .to_vec()
            .windows(window_size)
            .step_by(step_size)
            .enumerate()
        {
            let mut fft_input: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); fft_size];

            for (j, (&s, &w)) in window.iter().zip(hann_window.iter()).enumerate() {
                fft_input[j] = Complex32::new(s * w, 0.0);
            }

            fft.process(&mut fft_input);

            let fft_output: Array1<Complex32> = Array1::from_iter(
                fft_input
                    .iter()
                    .take(fft_size / 2 + 1)
                    .map(|&c| Complex32::new(c.re, c.im)),
            );

            stft_result.slice_mut(s![c, i, ..]).assign(&fft_output);
        }
    }

    stft_result.permuted_axes((0, 2, 1))
}

fn do_stft(
    audio: Array2<f32>,
    window_size: usize,
    step_size: usize,
    fft_size: usize,
) -> Array3<f32> {
    let stft_tensor = stft(audio, window_size, step_size, fft_size);
    let stft_tensor = stft_tensor.mapv(|a| a.re.powi(2) + a.im.powi(2));
    let stft_tensor = stft_tensor.mapv(|a| a.sqrt());
    stft_tensor
}

fn dot_with_filterbank_energies(
    audio: Array3<f32>,
    sampling_rate: f64,
    fft_size: usize,
    n_mels: usize,
    f_min: Option<f64>,
    f_max: Option<f64>,
) -> Array3<f32> {
    let filterbanks = mel(sampling_rate, fft_size, n_mels, f_min, f_max, false, true)
        .mapv(|a| a as f32)
        .insert_axis(Axis(0));

    let mut result: Array3<f32> =
        Array3::zeros((filterbanks.dim().0, filterbanks.dim().1, audio.dim().2));

    for i in 0..filterbanks.dim().0 {
        let a_2d: Array2<f32> = filterbanks.slice(s![i, .., ..]).to_owned();
        let b_2d: Array2<f32> = audio.slice(s![i, .., ..]).to_owned();
        let result_2d = a_2d.dot(&b_2d);

        result.slice_mut(s![i, .., ..]).assign(&result_2d);
    }

    result
}

fn normalize_per_feature(
    audio: Array3<f32>,
    seq_len: &Array1<usize>,
    constant: f32,
) -> (Array3<f32>, Array2<f32>, Array2<f32>) {
    let mut x_mean: Array2<f32> = Array2::zeros((seq_len.shape()[0], audio.shape()[1]));
    let mut x_std: Array2<f32> = Array2::zeros((seq_len.shape()[0], audio.shape()[1]));

    for i in 0..audio.shape()[0] {
        let third_dim = min(seq_len[i], audio.dim().2);
        if audio.slice(s![i, .., ..third_dim]).shape()[0] == 1 {
            panic!("normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result in torch.std() returning nan. Make sure your audio length has enough samples for a single feature (ex. at least `hop_length` for Mel Spectrograms).")
        }
        x_mean.slice_mut(s![i, ..]).assign(
            &audio
                .slice(s![i, .., ..min(third_dim, audio.dim().2)])
                .mean_axis(Axis(1))
                .unwrap(),
        );
        x_std
            .slice_mut(s![i, ..])
            .assign(&audio.slice(s![i, .., ..third_dim]).std_axis(Axis(1), 1.0));
    }
    let x_std = x_std.mapv(|a| a + constant);
    (
        (audio - x_mean.view().insert_axis(Axis(2))) / x_std.view().insert_axis(Axis(2)),
        x_mean,
        x_std,
    )
}

fn arange(start: f32, end: f32, step: f32) -> Array1<f32> {
    if step == 0.0 {
        panic!("Step cannot be zero");
    }

    let len = ((end - start) / step).ceil() as usize;
    Array1::from_shape_fn(len, |i| start + i as f32 * step)
}

fn repeat(array: Array1<f32>, repeats: (usize, usize)) -> Array2<f32> {
    let (repeat_rows, repeat_cols) = repeats;

    // Create a 2D array where each row is a repeated 1D array
    let rows: Vec<Vec<f32>> = (0..repeat_rows)
        .map(|_| array.to_vec()) // Repeat the 1D array to create rows
        .collect();

    // Flatten the 2D rows into a single vector and repeat each row
    let repeated_rows: Vec<f32> = rows
        .iter()
        .flat_map(|row| row.iter().flat_map(|&x| vec![x; repeat_cols]))
        .collect();

    // Create the final Array2
    Array2::from_shape_vec((repeat_rows, repeat_cols * array.len()), repeated_rows).unwrap()
}

fn run_mask(audio: Array3<f32>, seq_len: Array1<usize>) -> Array3<f32> {
    let max_len = audio.dim().2 as f32;
    let mask = arange(0.0, max_len, 1.0);
    let arr1 = repeat(mask, (audio.dim().0, 1));
    let arr2 = seq_len
        .insert_axis(Axis(1))
        .broadcast(arr1.dim())
        .unwrap()
        .mapv(|a| a as f32);

    let mut mask: Array2<bool> = Array2::from_elem(arr1.dim(), false);
    Zip::from(&mut mask)
        .and(&arr1)
        .and(&arr2)
        .for_each(|a, &b, &c| *a = b >= c);

    let mask = mask
        .insert_axis(Axis(1))
        .broadcast(audio.dim())
        .unwrap()
        .to_owned();
    let mut masked_signal = audio.clone();
    Zip::from(&mut masked_signal)
        .and(&audio)
        .and(&mask)
        .for_each(|a, &b, &c| *a = if c { 0.0 } else { b });

    masked_signal
}

fn get_argmax(arr: &Array3<f32>) -> Array2<usize> {
    let dims = arr.dim();

    let mut argmax: Array2<usize> = Array2::zeros((dims.0, dims.1));
    for batch in 0..dims.0 {
        for timestep in 0..dims.1 {
            argmax
                .slice_mut(s![batch, timestep])
                .fill(arr.slice(s![batch, timestep, ..]).argmax().unwrap());
        }
    }
    argmax
}

fn merge_logprobs(arr: &Array2<usize>) -> Vec<Vec<usize>> {
    let mut result = Vec::new();

    // Iterate over the slices along the specified axis
    for slice in arr.axis_iter(Axis(0)) {
        let mut unique = Vec::new();
        let mut previous = None;

        // Iterate over the elements of the current slice
        for &value in slice.iter() {
            if Some(value) != previous {
                unique.push(value);
                previous = Some(value);
            }
        }

        result.push(unique);
    }

    result
}

fn get_text(vocab: &Vec<String>, indices: Vec<usize>) -> String {
    let mut text = String::from("");
    for i in indices {
        let mut token = vocab[i].as_str();

        if token == "b" {
            token = "";
        }

        text += token;
    }

    text.replace("▁", " ")
}

fn array3_to_js_array(audio: Array3<f32>) -> JSArray {
    let audio_array: JSArray = JSArray::new();
    let (n, m, _) = audio.dim();
    for i in 0..n {
        let arr_2d: JSArray = JSArray::new();
        for j in 0..m {
            // Convert each 1D slice to Vec<T>
            let arr_1d: JSFloat32Array =
                JSFloat32Array::from(audio.slice(s![i, j, ..]).to_vec().as_slice());
            arr_2d.push(&arr_1d.into());
        }
        audio_array.push(&arr_2d.into());
    }
    audio_array
}

fn js_array_to_array3(
    arr: JSArray,
    shape: &[usize],
    vocab_start: usize,
    vocab_end: usize,
) -> Array3<f32> {
    let arr3 = Array3::from_shape_vec(
        (shape[0], shape[1], shape[2]),
        arr.to_vec()
            .iter()
            .map(|a| a.as_f64().unwrap() as f32)
            .collect(),
    )
    .unwrap();

    let logits_arr = arr3.slice(s![.., .., vocab_start..vocab_end]).to_owned();
    let blanks_arr = arr3.slice(s![.., .., (shape[2] - 1)..]).to_owned();
    concatenate(Axis(2), &[logits_arr.view(), blanks_arr.view()]).unwrap()
}

#[wasm_bindgen]
pub fn run_preprocessor(audio_file: &[u8]) -> JSArray {
    #[cfg(debug_assertions)]
    utils::set_panic_hook();
    let cursor = Cursor::new(audio_file.to_owned());
    let buf_reader: Box<dyn ReadSeek> = Box::new(BufReader::new(cursor));
    let wav: Wav<i16> = Wav::new(buf_reader).unwrap();
    let (audio_tensor, _): (Array2<i16>, i32) = wav.into_ndarray().unwrap();
    let audio_tensor = audio_tensor.t().to_owned();
    let seq_len = Array1::from_vec(vec![audio_tensor.ncols()]);
    let audio_tensor = audio_tensor.mapv(f32::from);
    let audio_tensor = audio_tensor.mapv(|a| a / 32768.0);
    let audio_tensor = do_premphasis(audio_tensor, 0.97);
    let audio_tensor = do_stft(
        audio_tensor,
        400, /*0.025 * 16000.0*/
        160, /*0.01 * 16000.0*/
        512,
    );
    let audio_tensor = audio_tensor.mapv(|a| a.powi(2));
    let audio_tensor =
        dot_with_filterbank_energies(audio_tensor, 16000.0, 512, 80, Some(0.0), None);
    let audio_tensor = audio_tensor.mapv(|a| a + 2_f32.powi(-24)).mapv(|a| a.ln());
    let (audio_tensor, _, _) = normalize_per_feature(audio_tensor, &seq_len, 1e-5);
    let audio_tensor = run_mask(audio_tensor, seq_len);

    array3_to_js_array(audio_tensor)
}

#[wasm_bindgen]
pub fn decode_logprobs(
    logprobs: JSArray,
    shape: &[usize],
    vocab_arr: JSArray,
    offset: usize,
    actual_vocab_size: usize,
) -> JSArray {
    let vocab_start = offset * actual_vocab_size;
    let vocab_end = vocab_start + actual_vocab_size;

    let arr = js_array_to_array3(logprobs, shape, vocab_start, vocab_end);

    let argmax = get_argmax(&arr);
    let indices_batch = merge_logprobs(&argmax);

    let mut vocab: Vec<String> = vocab_arr.to_vec()[vocab_start..vocab_end]
        .iter()
        .map(|a| a.as_string().unwrap())
        .collect();

    vocab.push(String::from("b"));

    let text: JSArray = JSArray::new();
    for indices in indices_batch {
        let t = get_text(&vocab, indices);
        text.push(&JsValue::from_str(&t.as_str().trim()));
    }

    text
}
