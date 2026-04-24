.PHONY: nested_Hello_World clean

nested_Hello_World:
	nvcc -arch=sm_90 nested_Hello_World.cu -o nested_Hello_World -lcudadevrt --relocatable-device-code true

clean:
	rm nested_Hello_World
