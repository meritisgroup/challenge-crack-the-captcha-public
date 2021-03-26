package captcha;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;

import com.github.marschall.memoryfilesystem.MemoryFileSystemBuilder;

public class MemoryFS {
	final static String dataPath = "datas"; // Path to data folder
	final static String modelPath = dataPath + "/models"; // Path to models folder
	final static String logsPath = dataPath + "/models/logs"; // Path to models folder
	final static String trainPath = dataPath + "/prepare"; // Path to train folder
	
	public static void main(String[] args) throws IOException {
		Path p = copyIntoMemory();
		
		System.out.println(Files.exists(p));
		
		Files.walkFileTree(p, new FileVisitor<Object>() {

			@Override
			public FileVisitResult preVisitDirectory(Object dir, BasicFileAttributes attrs) throws IOException {
				System.out.println(dir);
				return FileVisitResult.CONTINUE;
			}

			@Override
			public FileVisitResult visitFile(Object file, BasicFileAttributes attrs) throws IOException {
				System.out.println(file);
				return FileVisitResult.CONTINUE;
			}

			@Override
			public FileVisitResult visitFileFailed(Object file, IOException exc) throws IOException {
				return FileVisitResult.CONTINUE;
			}

			@Override
			public FileVisitResult postVisitDirectory(Object dir, IOException exc) throws IOException {
				return FileVisitResult.CONTINUE;
			}
		});
	}


	public static Path copyIntoMemory() throws IOException {
		File parentDir = new File(trainPath);
		Path samplesPath = parentDir.toPath();
		FileSystem fileSystem = MemoryFileSystemBuilder.newEmpty().build();
		Path p = fileSystem.getPath("p");
		Files.createDirectories(p);
		copyFolder(samplesPath, p);
		return p;
	}
	
	
	public static void copyFolder(Path src, Path dest) {
	    try {
	        Files.walk( src ).forEach( s -> {
	            try {
	                Path d = dest.resolve( src.relativize(s).toString().replace('\\', '/') );
	                if( Files.isDirectory( s ) ) {
	                    if( !Files.exists( d ) )
	                        Files.createDirectory( d );
	                    return;
	                }
	                Files.copy( s, d );// use flag to override existing
	            } catch( Exception e ) {
	                e.printStackTrace();
	            }
	        });
	    } catch( Exception ex ) {
	        ex.printStackTrace();
	    }
	}
}
