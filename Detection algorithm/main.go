package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/nfnt/resize"
)

var (
	model       *tf.SavedModel
	labels      []string
	graph       *tf.Graph
	session     *tf.Session
	inputHeight = 224
	inputWidth  = 224
)

func loadLabels(filename string) ([]string, error) {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(content), "\n")
	var filtered []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			filtered = append(filtered, line)
		}
	}
	return filtered, nil
}

func readImage(r io.Reader) (image.Image, error) {
	img, err := jpeg.Decode(r)
	if err == nil {
		return img, nil
	}
	_, err = r.Seek(0, io.SeekStart) 
	if seeker, ok := r.(io.Seeker); ok {
		seeker.Seek(0, io.SeekStart)
	}
	img, err = png.Decode(r)
	if err == nil {
		return img, nil
	}
	return nil, fmt.Errorf("unsupported image format (only JPEG or PNG supported)")
}

func preprocessImage(img image.Image) (*tf.Tensor, error) {
	resized := resize.Resize(uint(inputWidth), uint(inputHeight), img, resize.Lanczos3)

	bounds := resized.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	var imgData []float32

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			imgData = append(imgData, float32(r)/65535.0)
			imgData = append(imgData, float32(g)/65535.0)
			imgData = append(imgData, float32(b)/65535.0)
		}
	}
	tensor, err := tf.NewTensor([1]int{1})
	if err != nil {
		return nil, err
	}
	batch := make([][][][]float32, 1)
	batch[0] = make([][][]float32, height)
	idx := 0
	for y := 0; y < height; y++ {
		batch[0][y] = make([][]float32, width)
		for x := 0; x < width; x++ {
			p := make([]float32, 3)
			copy(p, imgData[idx:idx+3])
			idx += 3
			batch[0][y][x] = p
		}
	}

	tensor, err = tf.NewTensor(batch)
	if err != nil {
		return nil, err
	}

	return tensor, nil
}

func recognizeImage(tensor *tf.Tensor) ([]RecognitionResult, error) {
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			graph.Operation("MobilenetV1/Predictions/Reshape_1").Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, err
	}

	predictions := output[0].Value().([][]float32)[0]

	results := getTopK(predictions, 5)

	return results, nil
}

type RecognitionResult struct {
	Label      string  `json:"label"`
	Confidence float32 `json:"confidence"`
}

func getTopK(predictions []float32, k int) []RecognitionResult {
	type pred struct {
		index int
		value float32
	}
	var preds []pred
	for i, v := range predictions {
		preds = append(preds, pred{i, v})
	}
	for i := 0; i < len(preds)-1; i++ {
		for j := i + 1; j < len(preds); j++ {
			if preds[j].value > preds[i].value {
				preds[i], preds[j] = preds[j], preds[i]
			}
		}
	}
	if k > len(preds) {
		k = len(preds)
	}
	var results []RecognitionResult
	for i := 0; i < k; i++ {
		idx := preds[i].index
		label := "unknown"
		if idx < len(labels) {
			label = labels[idx]
		}
		results = append(results, RecognitionResult{
			Label:      label,
			Confidence: preds[i].value,
		})
	}
	return results
}

func recognizeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed, use POST", http.StatusMethodNotAllowed)
		return
	}
	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "Error reading image file: "+err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, err := readImage(file)
	if err != nil {
		http.Error(w, "Error decoding image: "+err.Error(), http.StatusBadRequest)
		return
	}

	inputTensor, err := preprocessImage(img)
	if err != nil {
		http.Error(w, "Error preprocessing image: "+err.Error(), http.StatusInternalServerError)
		return
	}

	results, err := recognizeImage(inputTensor)
	if err != nil {
		http.Error(w, "Error running inference: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func main() {
	modelPath := "mobilenet_v1_1.0_224_frozen.pb"
	labelsPath := "labels.txt"

	graphData, err := ioutil.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to read model file %q: %v", modelPath, err)
	}

	graph = tf.NewGraph()
	if err := graph.Import(graphData, ""); err != nil {
		log.Fatalf("Failed to import graph: %v", err)
	}

	session, err = tf.NewSession(graph, nil)
	if err != nil {
		log.Fatalf("Failed to create TensorFlow session: %v", err)
	}
	defer session.Close()

	labels, err = loadLabels(labelsPath)
	if err != nil {
		log.Fatalf("Failed to load labels: %v", err)
	}

	http.HandleFunc("/recognize", recognizeHandler)
	fmt.Println("Server started at :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}


