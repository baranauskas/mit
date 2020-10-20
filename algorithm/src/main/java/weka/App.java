/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.MIT;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Caíque Augusto Ferre
 */
public class App {

    public static void main(String[] args) throws Exception {

     

        // CONFIGURE THE ARFF FILE
        //String pathARFF = "C://Users//caique//Desktop//Teste//v2//datasets-20191127T172054Z-001//datasets//primary-tumor.arff";
        //String pathARFF = "C://Users//Caíque Augusto Ferre//Desktop//GIT//Mestrado//Datasets - Oficial//datasets-arff//8 - Bridges-Version-1//bridges.version1.arff";
        
        
        
        String pathARFF = "C://Users//Caíque Augusto Ferre//Desktop//GIT//Mestrado//mit//datasets//weather.numeric.arff";
        
        //String pathARFF = "C://Users//Caíque Augusto Ferre//Desktop//GIT//Mestrado//Datasets - Oficial//datasets-arff//1 - Anneal//anneal.arff";
        




        //String pathARFF = "C://Users//caique//Desktop//Computação Bioinspiada//dataset.arff";
        //String pathARFF = "src/main/java/MetaInductionTree/Core/Datasets/robot.arff";            
        //String pathARFF = "C://Users//caique//Desktop//Datasets UCI//Absenteeism_at_work_AAA//Absenteeism_at_work.arff";        

        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(pathARFF);

        Instances ins = ds.getDataSet();
        //ins.setClassIndex(20); // breast-cancer
        ins.setClassIndex(4); // weather        
        //ins.setClassIndex(38); // car
        
        
        
        
        Integer indexClass = ins.classAttribute().index();
        
        
        Attribute c = ins.classAttribute();
        Attribute b = ins.attribute(indexClass);
        AttributeStats a = ins.attributeStats(indexClass);            
      
        
        
        

//        Instances teste = new Instances(ins);
//        
//        teste.clear();
//
//        ins.randomize(ins.getRandomNumberGenerator(1));
//
//        List<Instance> a = ins.subList(0, 1000);
//        
//        teste.addAll(a);
//        
//        System.out.println(teste.toString());
        // CREATE THE CLASSIFIER
        MIT mit = new MIT();

        String j48Options = "-C 0.25 -M 2 -I 5 -STG A -W N"; //weka.classifiers.trees.MIT -C 0.25 -M 2 -I 100

        String[] optionsJ48 = j48Options.split("\\s", 0);

        mit.setOptions(optionsJ48);

        System.out.println("\nBegin build");

        mit.buildClassifier(ins);

        System.out.println("\n\n\n\n Classifier Normalized");

        System.out.println(mit.toString());

        System.out.println("\n\n\n\n Classifier Original");

        System.out.println(mit.toStringOriginal());

        System.out.println("\nEnd build");
//        
//        

//        
//        J48 j48 = new J48();
//
//        String j48Options2 = "-C 0.25 -M 2"; //weka.classifiers.trees.MIT -C 0.25 -M 2 -I 100
//
//        String[] optionsJ482 = j48Options2.split("\\s", 0);
//
//        j48.setOptions(optionsJ482);    
//        
//        j48.buildClassifier(ins);
//
//        Classifier cl = j48;
//        Evaluation eval = new Evaluation(ins);
//        eval.crossValidateModel(cl, ins, 10, new Random(1));
        // generate curve
//        ThresholdCurve tc = new ThresholdCurve();
//        int classIndex = 0;
//
//        Instances result = tc.getCurve(eval.predictions(), classIndex);
//        
//        double a = ThresholdCurve.getROCArea(result);
        //double b = eval.weightedAreaUnderROC();
//        System.out.println("\n\n");
//
//        System.out.println(mit.toString());
//
//        Evaluation evaluation = new Evaluation(ins);
//
//        Random Rnd = new Random(1);
//
//        evaluation.crossValidateModel(mit, ins, 3, Rnd);
//
//        System.out.println(evaluation.toSummaryString());
//
//        System.out.println("Deu certo");
    }
}


/*

LINKS UTEIS

https://stackoverflow.com/questions/19682996/datatable-to-html-table

*/
