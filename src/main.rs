use ndarray::{array, Array1, Array2};

fn purelin(x: Array1<f64>) -> Array1<f64> {
    x
}

fn poslin(x: Array1<f64>) -> Array1<f64> {
    x.mapv(|v| if v < 0.0 { 0.0} else { v })
}
fn compute_a1(w: &Array2<f64>, x:&Array1<f64>, b: &Array1<f64>) -> Array1<f64>{
    let wx = w.dot(x);
    let sum = &wx +b;
    purelin(sum)
}


fn main() {
    let t = array![
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ];

    println!("T matrix: \n{:#?}", t);
    let w = t.t().to_owned();
    println!("weight matrix: \n{:#?}", w);
    let b = array![3.0, 3.0];
    let p = array![1.0, 1.0, -1.0];


    let w2 = array![
        [1.0, -0.5],
        [-0.5, 1.0],
    ];
    let a1_output = compute_a1(&w, &p, &b);
    let mut a2_old = a1_output.clone();
    let mut a2_new = a2_old.clone();

    for _ in 0..100{
        a2_new = poslin(w2.dot(&a2_old));

        if a2_new == a2_old{
            break;
        }
        a2_old = a2_new.clone();

    }

    if a2_new[0] > 0.0{
        println!(" The input is classified as: Orange")
    }else{
        println!(" The input is classified as: Apple")
    }

}
