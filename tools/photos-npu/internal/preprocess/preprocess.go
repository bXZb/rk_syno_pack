package preprocess

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math"
)

// Config matches npu_model_conf.json network blocks.
type Config struct {
	Width   int
	Height  int
	Mean    float32
	RGBMean []float32
	Scale   float32
	Reorder bool // RGB -> BGR after sampling
	Stretch bool // distort to WxH instead of letterbox
}

// Result is a prepared NCHW tensor plus the geometry needed to map
// network-space boxes back onto the source image.
type Result struct {
	Data    []float32
	Width   int
	Height  int
	SrcW    int
	SrcH    int
	OffsetX int
	OffsetY int
	ScaleX  float64
	ScaleY  float64
}

func DecodeImage(buf []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

// NCHWFloat32 letterboxes the image into a WxH RGB NCHW float32 tensor:
// out[c*H*W + y*W + x] = (pixel[c] - mean) * scale
func NCHWFloat32(img image.Image, cfg Config) []float32 {
	return Prepare(img, cfg).Data
}

func Prepare(img image.Image, cfg Config) Result {
	w, h := cfg.Width, cfg.Height
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(dst, dst.Bounds(), image.Black, image.Point{}, draw.Src)

	res := Result{Width: w, Height: h, ScaleX: 1, ScaleY: 1}
	sb := img.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	res.SrcW, res.SrcH = sw, sh
	if sw <= 0 || sh <= 0 {
		res.Data = make([]float32, 3*w*h)
		return res
	}

	var ox, oy, nw, nh int
	if cfg.Stretch {
		nw, nh = w, h
		ox, oy = 0, 0
		res.ScaleX = float64(w) / float64(sw)
		res.ScaleY = float64(h) / float64(sh)
	} else {
		scale := math.Min(float64(w)/float64(sw), float64(h)/float64(sh))
		nw = int(math.Round(float64(sw) * scale))
		nh = int(math.Round(float64(sh) * scale))
		if nw < 1 {
			nw = 1
		}
		if nh < 1 {
			nh = 1
		}
		ox = (w - nw) / 2
		oy = (h - nh) / 2
		res.ScaleX, res.ScaleY = scale, scale
	}
	res.OffsetX, res.OffsetY = ox, oy
	resized := resizeNearest(img, nw, nh)
	draw.Draw(dst, image.Rect(ox, oy, ox+nw, oy+nh), resized, image.Point{}, draw.Src)

	meanR, meanG, meanB := cfg.Mean, cfg.Mean, cfg.Mean
	if len(cfg.RGBMean) >= 3 {
		meanR, meanG, meanB = cfg.RGBMean[0], cfg.RGBMean[1], cfg.RGBMean[2]
	} else if len(cfg.RGBMean) == 1 {
		meanR, meanG, meanB = cfg.RGBMean[0], cfg.RGBMean[0], cfg.RGBMean[0]
	}
	sc := cfg.Scale
	if sc == 0 {
		sc = 1
	}
	out := make([]float32, 3*w*h)
	plane := w * h
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := dst.RGBAAt(x, y).RGBA()
			rf := float32(r >> 8)
			gf := float32(g >> 8)
			bf := float32(b >> 8)
			if cfg.Reorder {
				rf, gf, bf = bf, gf, rf
			}
			i := y*w + x
			out[i] = (rf - meanR) * sc
			out[plane+i] = (gf - meanG) * sc
			out[2*plane+i] = (bf - meanB) * sc
		}
	}
	res.Data = out
	return res
}

// MapBox converts a box in network pixel space back to the source image.
func (r Result) MapBox(x1, y1, x2, y2 float32) (float32, float32, float32, float32) {
	return r.MapX(x1), r.MapY(y1), r.MapX(x2), r.MapY(y2)
}

func (r Result) MapX(x float32) float32 {
	if r.ScaleX == 0 {
		return x
	}
	v := (float64(x) - float64(r.OffsetX)) / r.ScaleX
	if v < 0 {
		v = 0
	}
	if r.SrcW > 0 && v > float64(r.SrcW) {
		v = float64(r.SrcW)
	}
	return float32(v)
}

func (r Result) MapY(y float32) float32 {
	if r.ScaleY == 0 {
		return y
	}
	v := (float64(y) - float64(r.OffsetY)) / r.ScaleY
	if v < 0 {
		v = 0
	}
	if r.SrcH > 0 && v > float64(r.SrcH) {
		v = float64(r.SrcH)
	}
	return float32(v)
}

func (r Result) MapPoints(xs, ys []float32) ([]float32, []float32) {
	ox := make([]float32, len(xs))
	oy := make([]float32, len(ys))
	for i := range xs {
		ox[i] = r.MapX(xs[i])
	}
	for i := range ys {
		oy[i] = r.MapY(ys[i])
	}
	return ox, oy
}

func resizeNearest(src image.Image, nw, nh int) *image.RGBA {
	sb := src.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	for y := 0; y < nh; y++ {
		sy := sb.Min.Y + y*sh/nh
		for x := 0; x < nw; x++ {
			sx := sb.Min.X + x*sw/nw
			dst.Set(x, y, src.At(sx, sy))
		}
	}
	return dst
}
