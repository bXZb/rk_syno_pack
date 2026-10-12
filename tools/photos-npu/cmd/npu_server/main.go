package main

import (
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/bXZb/rk_syno_pack/tools/photos-npu/internal/npupb"
	"github.com/bXZb/rk_syno_pack/tools/photos-npu/internal/server"
	"google.golang.org/grpc"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("npu_server: ")

	base, _ := os.Executable()
	defaultDir := filepath.Dir(base)
	if cwd, err := os.Getwd(); err == nil {
		if _, err := os.Stat(filepath.Join(cwd, "npu_model_conf.json")); err == nil {
			defaultDir = cwd
		}
	}

	listen := flag.String("listen", "unix:///run/synofoto/npu-photo.sock", "gRPC listen URI")
	dir := flag.String("dir", defaultDir, "npu working directory")
	flag.Parse()
	if flag.NArg() > 0 && strings.Contains(flag.Arg(0), "://") {
		*listen = flag.Arg(0)
	}

	network, addr := parseURI(*listen)
	if network == "unix" {
		if err := os.MkdirAll(filepath.Dir(addr), 0755); err != nil {
			log.Fatalf("mkdir: %v", err)
		}
		_ = os.Remove(addr)
	}

	svc, err := server.New(*dir)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	defer svc.Close()

	gs := grpc.NewServer()
	npupb.RegisterModelServer(gs, svc)

	lis, err := net.Listen(network, addr)
	if err != nil {
		log.Fatalf("listen %s: %v", *listen, err)
	}
	log.Printf("npu server listening on %s (dir=%s)", *listen, *dir)

	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
		<-ch
		gs.GracefulStop()
	}()
	if err := gs.Serve(lis); err != nil {
		log.Fatal(err)
	}
}

func parseURI(uri string) (network, addr string) {
	switch {
	case strings.HasPrefix(uri, "unix://"):
		return "unix", strings.TrimPrefix(uri, "unix://")
	case strings.HasPrefix(uri, "tcp://"):
		return "tcp", strings.TrimPrefix(uri, "tcp://")
	case strings.HasPrefix(uri, "/"):
		return "unix", uri
	default:
		return "tcp", uri
	}
}
