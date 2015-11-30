
class JadamiluTest {
	
    private static native int pjd( /** args... */ );

    public static void main(String args[]) {
        System.out.println("-- We are in the Java program --");
        
		pjd();
		
        System.out.println("Exit Java");
    }

    static {
        // Call up the static library. There should be a "libjadamilu.so"
	// in the working directory to have this properly run.
        System.loadLibrary("jadamilu");
    }
}