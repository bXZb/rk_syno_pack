package server

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"net"
	"os"
	"path/filepath"
	"testing"

	"github.com/bXZb/rk_syno_pack/tools/photos-npu/internal/npupb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func startTestServer(t *testing.T) npupb.ModelClient {
	t.Helper()
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "asset", "labels.txt"), "cat\ndog\n")
	mustWrite(t, filepath.Join(dir, "asset", "thresholds.txt"), "0.1\n0.1\n")
	svc, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(svc.Close)

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	npupb.RegisterModelServer(gs, svc)
	go gs.Serve(lis)
	t.Cleanup(gs.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return npupb.NewModelClient(conn)
}

func mustWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestExecCmdReady(t *testing.T) {
	c := startTestServer(t)
	rep, err := c.ExecCmd(context.Background(), &npupb.CmdRequest{Name: "status"})
	if err != nil {
		t.Fatal(err)
	}
	if rep.GetResult() != "ready" {
		t.Fatalf("result=%q", rep.GetResult())
	}
}

func TestConceptDetectEmptyWithoutRuntime(t *testing.T) {
	c := startTestServer(t)
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		t.Fatal(err)
	}
	rep, err := c.ConceptDetectBuffer(context.Background(), &npupb.BufferRequest{Buffer: buf.Bytes()})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.GetReplyMap()) != 0 {
		t.Fatalf("expected empty map, got %v", rep.GetReplyMap())
	}
}

func TestFaceDetectEmptyWithoutRuntime(t *testing.T) {
	c := startTestServer(t)
	img := image.NewRGBA(image.Rect(0, 0, 32, 32))
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		t.Fatal(err)
	}
	rep, err := c.FaceDetectBuffer(context.Background(), &npupb.BufferRequest{Buffer: buf.Bytes()})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.GetFaceInfo()) != 0 {
		t.Fatalf("unexpected faces: %d", len(rep.GetFaceInfo()))
	}
}
