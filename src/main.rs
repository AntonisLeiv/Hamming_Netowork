use ndarray::{array, Array1, Array2};
/// This is a simple implementation of a Hamming network.
/// we have to layers. The first layer has 2 neurons and so does the second 
/// We try to identify apples vs  oranges
/// the input of the network is given below with the varible name p


///Linear activation function: f(x) = x
fn purelin(x: Array1<f64>) -> Array1<f64> {
    x
}
///Positive linear function f(x) = (x : if x > 0 | 0 if x < 0)
/// This is used as the activation function in the second layer.
fn poslin(x: Array1<f64>) -> Array1<f64> {
    x.mapv(|v| if v < 0.0 { 0.0} else { v })
}

/// Compute activations of the first layer: a1 = purelin(W * x + b)
/// this is the output of the first layer
fn compute_a1(w: &Array2<f64>, x:&Array1<f64>, b: &Array1<f64>) -> Array1<f64>{
    let wx = w.dot(x);
    let sum = &wx +b;
    purelin(sum)
}

/// 
fn main() {
    //Training vectors
    // each column is the prototype for and apple and orange
    // the first column is the orange
    // the second column is the apple
    // meaning of the varibles in each column = [shape, texture, weight]
    //shape = (1 if is round| -1 if is eliptical)
    // texture = (1 if is smooth| -1 if rough)
    // wight = (1 if is more than one pound| -1 if is less than one pound)
    let t = array![
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
    ];

    println!("T matrix: \n{:#?}", t);

    //Weight matrix is the transpose of the training vectors
    let w = t.t().to_owned();
    println!("weight matrix: \n{:#?}", w);

    //The bias
    let b = array![3.0, 3.0];
    
    //The input we give: lets say :
    // is almost round 0.85 
    // is pretty smooth 0.9
    // and is less than 1 pund
    let p = array![0.85, 0.9, -1.0];

    // weight matrix of the second layer
    let w2 = array![
        [1.0, -0.5],
        [-0.5, 1.0],
    ];

    // Here we start the second layer 
    let a1_output = compute_a1(&w, &p, &b);
    let mut a2_old = a1_output.clone();
    let mut a2_new = a2_old.clone();
    // This loop represents the layer
    for _ in 0..100{
        a2_new = poslin(w2.dot(&a2_old));

        if a2_new == a2_old{
            break;
        }
        a2_old = a2_new.clone();

    }

    // As output we will get a 2x1 vector 
    if a2_new[0] > 0.0{
        println!(" The input is classified as: Orange")
    }else{
        println!(" The input is classified as: Apple")
    }

}
