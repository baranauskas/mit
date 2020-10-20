/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 /*
 *    J48.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.classifiers.trees;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Optional;
import java.util.Stack;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import weka.classifiers.AbstractClassifier;
import static weka.classifiers.AbstractClassifier.runClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Drawable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matchable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.PartitionGenerator;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.classifiers.trees.j48.BinC45ModelSelection;
import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.PruneableClassifierTree;
import weka.core.AttributeStats;

/**
 * <!-- globalinfo-start --> Class for generating a pruned or unpruned C4.5
 * decision tree. For more information, see<br/>
 * <br/>
 * Ross Quinlan (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann
 * Publishers, San Mateo, CA.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#64;book{Quinlan1993,
 *    address = {San Mateo, CA},
 *    author = {Ross Quinlan},
 *    publisher = {Morgan Kaufmann Publishers},
 *    title = {C4.5: Programs for Machine Learning},
 *    year = {1993}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre>
 * -U
 *  Use unpruned tree.
 * </pre>
 *
 * <pre>
 * -O
 *  Do not collapse tree.
 * </pre>
 *
 * <pre>
 * -C &lt;pruning confidence&gt;
 *  Set confidence threshold for pruning.
 *  (default 0.25)
 * </pre>
 *
 * <pre>
 * -M &lt;minimum number of instances&gt;
 *  Set minimum number of instances per leaf.
 *  (default 2)
 * </pre>
 *
 * <pre>
 * -R
 *  Use reduced error pruning.
 * </pre>
 *
 * <pre>
 * -N &lt;number of folds&gt;
 *  Set number of folds for reduced error
 *  pruning. One fold is used as pruning set.
 *  (default 3)
 * </pre>
 *
 * <pre>
 * -B
 *  Use binary splits only.
 * </pre>
 *
 * <pre>
 * -S
 *  Don't perform subtree raising.
 * </pre>
 *
 * <pre>
 * -L
 *  Do not clean up after the tree has been built.
 * </pre>
 *
 * <pre>
 * -A
 *  Laplace smoothing for predicted probabilities.
 * </pre>
 *
 * <pre>
 * -J
 *  Do not use MDL correction for info gain on numeric attributes.
 * </pre>
 *
 * <pre>
 * -Q &lt;seed&gt;
 *  Seed for random data shuffling (default 1).
 * </pre>
 *
 * <pre>
 * -doNotMakeSplitPointActualValue
 *  Do not make split point actual value.
 * </pre>
 *
 * <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 14534 $
 */
public class MIT extends AbstractClassifier implements OptionHandler, Drawable,
        Matchable, Sourcable, WeightedInstancesHandler, Summarizable,
        AdditionalMeasureProducer, TechnicalInformationHandler, PartitionGenerator {

    /**
     * **************************************************************************************************************
     * EXTRACTED FROM J48 CLASS
     * **************************************************************************************************************
     */
    /**
     * for serialization
     */
    static final long serialVersionUID = -217733168393644444L;

    /**
     * The decision tree
     */
    protected ClassifierTree m_root;

    /**
     * Unpruned tree?
     */
    protected boolean m_unpruned = false;

    /**
     * Collapse tree?
     */
    protected boolean m_collapseTree = true;

    /**
     * Confidence level
     */
    protected float m_CF = 0.25f;

    /**
     * Minimum number of instances
     */
    protected int m_minNumObj = 2;

    /**
     * Use MDL correction?
     */
    protected boolean m_useMDLcorrection = true;

    /**
     * Determines whether probabilities are smoothed using Laplace correction
     * when predictions are generated
     */
    protected boolean m_useLaplace = false;

    /**
     * Use reduced error pruning?
     */
    protected boolean m_reducedErrorPruning = false;

    /**
     * Number of folds for reduced error pruning.
     */
    protected int m_numFolds = 3;

    /**
     * Binary splits on nominal attributes?
     */
    protected boolean m_binarySplits = false;

    /**
     * Subtree raising to be performed?
     */
    protected boolean m_subtreeRaising = true;

    /**
     * Cleanup after the tree has been built.
     */
    protected boolean m_noCleanup = false;

    /**
     * Random number seed for reduced-error pruning.
     */
    protected int m_Seed = 1;

    /**
     * Do not relocate split point to actual data value
     */
    protected boolean m_doNotMakeSplitPointActualValue;

    /**
     * Classifies an instance.
     *
     * @param instance the instance to classify
     * @return the classification for the instance
     * @throws Exception if instance can't be classified successfully
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {

        return m_root.classifyInstance(instance);
    }

    /**
     * Returns class probabilities for an instance.
     *
     * @param instance the instance to calculate the class probabilities for
     * @return the class probabilities
     * @throws Exception if distribution can't be computed successfully
     */
    @Override
    public final double[] distributionForInstance(Instance instance)
            throws Exception {

        return m_root.distributionForInstance(instance, m_useLaplace);
    }

    /**
     * Returns the type of graph this classifier represents.
     *
     * @return Drawable.TREE
     */
    @Override
    public int graphType() {
        return Drawable.TREE;
    }

    /**
     * Returns graph describing the tree.
     *
     * @return the graph describing the tree
     * @throws Exception if graph can't be computed
     */
    @Override
    public String graph() throws Exception {

        return m_root.graph();
    }

    public String graphNormalized() throws Exception {

        return m_root.graphNormalized(m_forestSize);
    }

    /**
     * Returns tree in prefix order.
     *
     * @return the tree in prefix order
     * @throws Exception if something goes wrong
     */
    @Override
    public String prefix() throws Exception {

        return m_root.prefix();
    }

    /**
     * Returns tree as an if-then statement.
     *
     * @param className the name of the Java class
     * @return the tree as a Java if-then type statement
     * @throws Exception if something goes wrong
     */
    @Override
    public String toSource(String className) throws Exception {

        StringBuffer[] source = m_root.toSource(className);
        return "class " + className + " {\n\n"
                + "  public static double classify(Object[] i)\n"
                + "    throws Exception {\n\n" + "    double p = Double.NaN;\n"
                + source[0] // Assignment code
                + "    return p;\n" + "  }\n" + source[1] // Support code
                + "}\n";
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String seedTipText() {
        return "The seed used for randomizing the data "
                + "when reduced-error pruning is used.";
    }

    /**
     * Get the value of Seed.
     *
     * @return Value of Seed.
     */
    public int getSeed() {

        return m_Seed;
    }

    /**
     * Set the value of Seed.
     *
     * @param newSeed Value to assign to Seed.
     */
    public void setSeed(int newSeed) {

        m_Seed = newSeed;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String useLaplaceTipText() {
        return "Whether counts at leaves are smoothed based on Laplace.";
    }

    /**
     * Get the value of useLaplace.
     *
     * @return Value of useLaplace.
     */
    public boolean getUseLaplace() {

        return m_useLaplace;
    }

    /**
     * Set the value of useLaplace.
     *
     * @param newuseLaplace Value to assign to useLaplace.
     */
    public void setUseLaplace(boolean newuseLaplace) {

        m_useLaplace = newuseLaplace;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String useMDLcorrectionTipText() {
        return "Whether MDL correction is used when finding splits on numeric attributes.";
    }

    /**
     * Get the value of useMDLcorrection.
     *
     * @return Value of useMDLcorrection.
     */
    public boolean getUseMDLcorrection() {

        return m_useMDLcorrection;
    }

    /**
     * Set the value of useMDLcorrection.
     *
     * @param newuseMDLcorrection Value to assign to useMDLcorrection.
     */
    public void setUseMDLcorrection(boolean newuseMDLcorrection) {

        m_useMDLcorrection = newuseMDLcorrection;
    }

    /**
     * Returns a superconcise version of the model
     *
     * @return a summary of the model
     */
    @Override
    public String toSummaryString() {

        return "Number of leaves: " + m_root.numLeaves() + "\n"
                + "Size of the tree: " + m_root.numNodes() + "\n";
    }

    /**
     * Returns the size of the tree
     *
     * @return the size of the tree
     */
    public double measureTreeSize() {
        return m_root.numNodes();
    }

    /**
     * Returns the number of leaves
     *
     * @return the number of leaves
     */
    public double measureNumLeaves() {
        return m_root.numLeaves();
    }

    /**
     * Returns the number of rules (same as number of leaves)
     *
     * @return the number of rules
     */
    public double measureNumRules() {
        return m_root.numLeaves();
    }

    /**
     * Returns an enumeration of the additional measure names
     *
     * @return an enumeration of the measure names
     */
    @Override
    public Enumeration<String> enumerateMeasures() {
        Vector<String> newVector = new Vector<String>(3);
        newVector.addElement("measureTreeSize");
        newVector.addElement("measureNumLeaves");
        newVector.addElement("measureNumRules");
        return newVector.elements();
    }

    /**
     * Returns the value of the named measure
     *
     * @param additionalMeasureName the name of the measure to query for its
     * value
     * @return the value of the named measure
     * @throws IllegalArgumentException if the named measure is not supported
     */
    @Override
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumRules") == 0) {
            return measureNumRules();
        } else if (additionalMeasureName.compareToIgnoreCase("measureTreeSize") == 0) {
            return measureTreeSize();
        } else if (additionalMeasureName.compareToIgnoreCase("measureNumLeaves") == 0) {
            return measureNumLeaves();
        } else {
            throw new IllegalArgumentException(additionalMeasureName
                    + " not supported (MIT)");
        }
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String unprunedTipText() {
        return "Whether pruning is performed.";
    }

    /**
     * Get the value of unpruned.
     *
     * @return Value of unpruned.
     */
    public boolean getUnpruned() {

        return m_unpruned;
    }

    /**
     * Set the value of unpruned. Turns reduced-error pruning off if set.
     *
     * @param v Value to assign to unpruned.
     */
    public void setUnpruned(boolean v) {

        if (v) {
            m_reducedErrorPruning = false;
        }
        m_unpruned = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String collapseTreeTipText() {
        return "Whether parts are removed that do not reduce training error.";
    }

    /**
     * Get the value of collapseTree.
     *
     * @return Value of collapseTree.
     */
    public boolean getCollapseTree() {

        return m_collapseTree;
    }

    /**
     * Set the value of collapseTree.
     *
     * @param v Value to assign to collapseTree.
     */
    public void setCollapseTree(boolean v) {

        m_collapseTree = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String confidenceFactorTipText() {
        return "The confidence factor used for pruning (smaller values incur "
                + "more pruning).";
    }

    /**
     * Get the value of CF.
     *
     * @return Value of CF.
     */
    public float getConfidenceFactor() {

        return m_CF;
    }

    /**
     * Set the value of CF.
     *
     * @param v Value to assign to CF.
     */
    public void setConfidenceFactor(float v) {

        m_CF = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String minNumObjTipText() {
        return "The minimum number of instances per leaf.";
    }

    /**
     * Get the value of minNumObj.
     *
     * @return Value of minNumObj.
     */
    public int getMinNumObj() {

        return m_minNumObj;
    }

    /**
     * Set the value of minNumObj.
     *
     * @param v Value to assign to minNumObj.
     */
    public void setMinNumObj(int v) {

        m_minNumObj = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String reducedErrorPruningTipText() {
        return "Whether reduced-error pruning is used instead of C.4.5 pruning.";
    }

    /**
     * Get the value of reducedErrorPruning.
     *
     * @return Value of reducedErrorPruning.
     */
    public boolean getReducedErrorPruning() {

        return m_reducedErrorPruning;
    }

    /**
     * Set the value of reducedErrorPruning. Turns unpruned trees off if set.
     *
     * @param v Value to assign to reducedErrorPruning.
     */
    public void setReducedErrorPruning(boolean v) {

        if (v) {
            m_unpruned = false;
        }
        m_reducedErrorPruning = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String numFoldsTipText() {
        return "Determines the amount of data used for reduced-error pruning. "
                + " One fold is used for pruning, the rest for growing the tree.";
    }

    /**
     * Get the value of numFolds.
     *
     * @return Value of numFolds.
     */
    public int getNumFolds() {

        return m_numFolds;
    }

    /**
     * Set the value of numFolds.
     *
     * @param v Value to assign to numFolds.
     */
    public void setNumFolds(int v) {

        m_numFolds = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String binarySplitsTipText() {
        return "Whether to use binary splits on nominal attributes when "
                + "building the trees.";
    }

    /**
     * Get the value of binarySplits.
     *
     * @return Value of binarySplits.
     */
    public boolean getBinarySplits() {

        return m_binarySplits;
    }

    /**
     * Set the value of binarySplits.
     *
     * @param v Value to assign to binarySplits.
     */
    public void setBinarySplits(boolean v) {

        m_binarySplits = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String subtreeRaisingTipText() {
        return "Whether to consider the subtree raising operation when pruning.";
    }

    /**
     * Get the value of subtreeRaising.
     *
     * @return Value of subtreeRaising.
     */
    public boolean getSubtreeRaising() {

        return m_subtreeRaising;
    }

    /**
     * Set the value of subtreeRaising.
     *
     * @param v Value to assign to subtreeRaising.
     */
    public void setSubtreeRaising(boolean v) {

        m_subtreeRaising = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String saveInstanceDataTipText() {
        return "Whether to save the training data for visualization.";
    }

    /**
     * Check whether instance data is to be saved.
     *
     * @return true if instance data is saved
     */
    public boolean getSaveInstanceData() {

        return m_noCleanup;
    }

    /**
     * Set whether instance data is to be saved.
     *
     * @param v true if instance data is to be saved
     */
    public void setSaveInstanceData(boolean v) {

        m_noCleanup = v;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String doNotMakeSplitPointActualValueTipText() {
        return "If true, the split point is not relocated to an actual data value."
                + " This can yield substantial speed-ups for large datasets with numeric attributes.";
    }

    /**
     * Gets the value of doNotMakeSplitPointActualValue.
     *
     * @return the value
     */
    public boolean getDoNotMakeSplitPointActualValue() {
        return m_doNotMakeSplitPointActualValue;
    }

    /**
     * Sets the value of doNotMakeSplitPointActualValue.
     *
     * @param m_doNotMakeSplitPointActualValue the value to set
     */
    public void setDoNotMakeSplitPointActualValue(
            boolean m_doNotMakeSplitPointActualValue) {
        this.m_doNotMakeSplitPointActualValue = m_doNotMakeSplitPointActualValue;
    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 14534 $");
    }

    /**
     * Builds the classifier to generate a partition.
     */
    @Override
    public void generatePartition(Instances data) throws Exception {

        buildClassifier(data);
    }

    /**
     * Computes an array that indicates node membership.
     */
    @Override
    public double[] getMembershipValues(Instance inst) throws Exception {

        return m_root.getMembershipValues(inst);
    }

    /**
     * Returns the number of elements in the partition.
     */
    @Override
    public int numElements() throws Exception {

        return m_root.numNodes();
    }

    /**
     * Main method for testing this class
     *
     * @param argv the commandline options
     */
    public static void main(String[] argv) {
        runClassifier(new MIT(), argv);
    }

    /**
     * **************************************************************************************************************
     * **************************************************************************************************************
     * **************************************************************************************************************
     * MODIFICATIONS OF J48 CLASS
     * ***********************************************************************************
     * **************************************************************************************************************
     * **************************************************************************************************************
     * **************************************************************************************************************
     */
    protected Instances m_metaInstances = null;

    protected int m_forestSize = 100;

    protected String m_strategyNumericAtts = "I";

    protected String m_strategyWeight = "P";

//    protected double m_errorRateAccepted = 1.0;
    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {

        return "MIT - Meta Induction Tree. For more "
                + "information, see\n\n" + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.BOOK);
        result.setValue(Field.AUTHOR, "Caíque Ferreira");
        result.setValue(Field.YEAR, "2019");
        result.setValue(Field.TITLE, "MIT: Meta Induction Tree");
        result.setValue(Field.PUBLISHER, "--Publisher--");
        result.setValue(Field.ADDRESS, "RP-SP, BR");

        return result;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result;

        result = new Capabilities(this);
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        // NOT YET
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        //result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Generates the classifier.
     *
     * @param instances the data to train the classifier with
     * @throws Exception if classifier can't be built successfully
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {

        Boolean printAnalysis = true;

        RandomForest rf = new RandomForest();

        rf.setNumIterations(m_forestSize); //100

        rf.buildClassifier(instances);

        Classifier[] randomForestClassifiers = rf.getClassifiers();

        // CREATE META TRAINING DATA 
        Instances metaTrainingData = new Instances(instances);

        metaTrainingData.clear();

        RandomTree randomTree = null;
        RandomTree.Tree tree = null;
        TreeAsRootToLeaf treeAsRootToLeaf = null;
        TreeAsARFF treeAsARFF = null;
        Instances instancesItem = null;

        // TAXA DE ERRO ACEITA, UTILIZADA NA SELEÇÃO DAS ÁRVORES. VALOR PADRÃO 1 (100%), OU SEJA, TRAZ TODAS. 
//        Double errorAcepted = m_errorRateAccepted * instances.sumOfWeights(); // avaliar no futuro, pois pode ser o instances.size()
        for (int i = 0; i < randomForestClassifiers.length; i++) {

            randomTree = (RandomTree) randomForestClassifiers[i];

            tree = (RandomTree.Tree) randomTree.getTree();

            treeAsRootToLeaf = new TreeAsRootToLeaf(tree, m_forestSize, instances);

//            if (treeAsRootToLeaf.getTreeError() > errorAcepted) {
//                continue;
//            }
            treeAsARFF = new TreeAsARFF(treeAsRootToLeaf, instances, m_strategyNumericAtts, m_strategyWeight);

            instancesItem = treeAsARFF.getInstancesMeta();

            metaTrainingData.addAll(instancesItem);

            if (printAnalysis) {

                TreeAsTable treeAsTable = new TreeAsTable(treeAsRootToLeaf, instances);

                System.out.print("\n\n\n\n\n\n\n\n------------------------------");

                System.out.print("\n\n--Dataset Name--\n");

                System.out.print(instances.relationName());

                System.out.print("\n\n--Dataset Num of Attrs--\n");

                System.out.print(instances.numAttributes());

                System.out.print("\n\n--Dataset Num of Classes--\n");

                System.out.print(instances.numClasses());

                System.out.print("\n\n--Dataset Num of Instances--\n");

                System.out.print(instances.numInstances());

                System.out.print("\n\n--Numeric Attributes--\n");

                List<Attribute> atts = Collections.list(instances.enumerateAttributes());

                for (Attribute att : atts) {
                    if (att.isNumeric()) {
                        System.out.print("\n" + att.name() + " -> Min: " + instances.attributeStats(att.index()).numericStats.min + " - Max: " + instances.attributeStats(att.index()).numericStats.max + "\n");
                    }
                }

                System.out.print("\n------------------------------\n\n\n");

                System.out.print("\n\n--Strategy--\n");

                System.out.print(m_strategyNumericAtts);

                System.out.print("\n\n-- Weight Strategy--\n");

                System.out.print(m_strategyWeight);

                System.out.print("\n\n--Tree--\n");

                System.out.print(randomTree.toString());

                System.out.print("\n\n--Tree as Root to Leaf--\n");

                System.out.print(treeAsRootToLeaf.toString());

                System.out.print("\n\n--Tree as Decision Table--\n");

                System.out.print(treeAsTable.toString());

                System.out.print("\n\n--Tree as ARFF--\n");

                System.out.print(treeAsARFF.toString());

                System.out.print("\n\n--Error Rate--\n");
//                System.out.print("\n\n Error Rate Accepted (Dataset): " + errorAcepted);
                System.out.print("\n\n Error of Tree: " + treeAsRootToLeaf.getTreeError().toString());

//                if (treeAsRootToLeaf.getTreeError() > errorAcepted) {
//
//                    System.out.print("\n Árvore foi excluída\n");
//                }
            }
        }

        if (printAnalysis) {
            System.out.print("\n\n\n\n\n\n--Meta Training Data--\n");

            System.out.println(metaTrainingData.toString());

            System.out.print("\n\n\n\n\n\n");

        }

        getCapabilities().testWithFail(metaTrainingData);

        ModelSelection modSelection;

        if (m_binarySplits) {
            modSelection = new BinC45ModelSelection(m_minNumObj, metaTrainingData,
                    m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
        } else {
            modSelection = new C45ModelSelection(m_minNumObj, metaTrainingData,
                    m_useMDLcorrection, m_doNotMakeSplitPointActualValue);
        }
        if (!m_reducedErrorPruning) {
            m_root = new C45PruneableClassifierTree(modSelection, !m_unpruned, m_CF,
                    m_subtreeRaising, !m_noCleanup, m_collapseTree);
        } else {
            m_root = new PruneableClassifierTree(modSelection, !m_unpruned,
                    m_numFolds, !m_noCleanup, m_Seed);
        }
        m_root.buildClassifier(metaTrainingData);
        if (m_binarySplits) {
            ((BinC45ModelSelection) modSelection).cleanup();
        } else {
            ((C45ModelSelection) modSelection).cleanup();
        }
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * Valid options are:
     * <p>
     *
     * -U <br>
     * Use unpruned tree.
     * <p>
     *
     * -C confidence <br>
     * Set confidence threshold for pruning. (Default: 0.25)
     * <p>
     *
     * -M number <br>
     * Set minimum number of instances per leaf. (Default: 2)
     * <p>
     *
     * -R <br>
     * Use reduced error pruning. No subtree raising is performed.
     * <p>
     *
     * -N number <br>
     * Set number of folds for reduced error pruning. One fold is used as the
     * pruning set. (Default: 3)
     * <p>
     *
     * -B <br>
     * Use binary splits for nominal attributes.
     * <p>
     *
     * -S <br>
     * Don't perform subtree raising.
     * <p>
     *
     * -L <br>
     * Do not clean up after the tree has been built.
     *
     * -A <br>
     * If set, Laplace smoothing is used for predicted probabilites.
     * <p>
     *
     * -Q <br>
     * The seed for reduced-error pruning.
     * <p>
     *
     * -I <br>
     * Forest Size
     * <p>
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>(13);

        newVector.addElement(new Option("\tUse unpruned tree.", "U", 0, "-U"));
        newVector.addElement(new Option("\tDo not collapse tree.", "O", 0, "-O"));
        newVector.addElement(new Option("\tSet confidence threshold for pruning.\n"
                + "\t(default 0.25)", "C", 1, "-C <pruning confidence>"));
        newVector.addElement(new Option(
                "\tSet minimum number of instances per leaf.\n" + "\t(default 2)", "M",
                1, "-M <minimum number of instances>"));
        newVector.addElement(new Option("\tUse reduced error pruning.", "R", 0,
                "-R"));
        newVector.addElement(new Option("\tSet number of folds for reduced error\n"
                + "\tpruning. One fold is used as pruning set.\n" + "\t(default 3)", "N",
                1, "-N <number of folds>"));
        newVector.addElement(new Option("\tUse binary splits only.", "B", 0, "-B"));
        newVector.addElement(new Option("\tDo not perform subtree raising.", "S", 0,
                "-S"));
        newVector.addElement(new Option(
                "\tDo not clean up after the tree has been built.", "L", 0, "-L"));
        newVector.addElement(new Option(
                "\tLaplace smoothing for predicted probabilities.", "A", 0, "-A"));
        newVector.addElement(new Option(
                "\tDo not use MDL correction for info gain on numeric attributes.", "J",
                0, "-J"));
        newVector.addElement(new Option(
                "\tSeed for random data shuffling (default 1).", "Q", 1, "-Q <seed>"));
        newVector.addElement(new Option("\tDo not make split point actual value.",
                "-doNotMakeSplitPointActualValue", 0, "-doNotMakeSplitPointActualValue"));

        // NEW PARAMETER
        newVector.addElement(new Option("\tForest Size.", "I", 0, "-I"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
     *
     * <!-- options-start --> Valid options are:
     * <p/>
     *
     * <pre>
     * -U
     *  Use unpruned tree.
     * </pre>
     *
     * <pre>
     * -O
     *  Do not collapse tree.
     * </pre>
     *
     * <pre>
     * -C &lt;pruning confidence&gt;
     *  Set confidence threshold for pruning.
     *  (default 0.25)
     * </pre>
     *
     * <pre>
     * -M &lt;minimum number of instances&gt;
     *  Set minimum number of instances per leaf.
     *  (default 2)
     * </pre>
     *
     * <pre>
     * -R
     *  Use reduced error pruning.
     * </pre>
     *
     * <pre>
     * -N &lt;number of folds&gt;
     *  Set number of folds for reduced error
     *  pruning. One fold is used as pruning set.
     *  (default 3)
     * </pre>
     *
     * <pre>
     * -B
     *  Use binary splits only.
     * </pre>
     *
     * <pre>
     * -S
     *  Don't perform subtree raising.
     * </pre>
     *
     * <pre>
     * -L
     *  Do not clean up after the tree has been built.
     * </pre>
     *
     * <pre>
     * -A
     *  Laplace smoothing for predicted probabilities.
     * </pre>
     *
     * <pre>
     * -J
     *  Do not use MDL correction for info gain on numeric attributes.
     * </pre>
     *
     * <pre>
     * -Q &lt;seed&gt;
     *  Seed for random data shuffling (default 1).
     * </pre>
     *
     * <pre>
     * -doNotMakeSplitPointActualValue
     *  Do not make split point actual value.
     * </pre>
     *
     * <pre>
     * -I
     *  Forest Size.
     * </pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        // Other options
        String minNumString = Utils.getOption('M', options);
        if (minNumString.length() != 0) {
            m_minNumObj = Integer.parseInt(minNumString);
        } else {
            m_minNumObj = 2;
        }

        // NEW PARAMETERS *****************************************
        String forestSizeString = Utils.getOption('I', options);
        if (forestSizeString.length() != 0) {
            m_forestSize = Integer.parseInt(forestSizeString);
        } else {
            m_forestSize = 15;
        }

        m_strategyNumericAtts = Utils.getOption("STG", options);
        m_strategyWeight = Utils.getOption("W", options);

//        m_errorRateAccepted = Double.parseDouble(Utils.getOption("ERA", options));
        // *****************************************       
        m_binarySplits = Utils.getFlag('B', options);
        m_useLaplace = Utils.getFlag('A', options);
        m_useMDLcorrection = !Utils.getFlag('J', options);

        // Pruning options
        m_unpruned = Utils.getFlag('U', options);
        m_collapseTree = !Utils.getFlag('O', options);
        m_subtreeRaising = !Utils.getFlag('S', options);
        m_noCleanup = Utils.getFlag('L', options);
        m_doNotMakeSplitPointActualValue = Utils.getFlag(
                "doNotMakeSplitPointActualValue", options);
        if ((m_unpruned) && (!m_subtreeRaising)) {
            throw new Exception(
                    "Subtree raising doesn't need to be unset for unpruned tree!");
        }
        m_reducedErrorPruning = Utils.getFlag('R', options);
        if ((m_unpruned) && (m_reducedErrorPruning)) {
            throw new Exception(
                    "Unpruned tree and reduced error pruning can't be selected "
                    + "simultaneously!");
        }
        String confidenceString = Utils.getOption('C', options);
        if (confidenceString.length() != 0) {
            if (m_reducedErrorPruning) {
                throw new Exception("Setting the confidence doesn't make sense "
                        + "for reduced error pruning.");
            } else if (m_unpruned) {
                throw new Exception(
                        "Doesn't make sense to change confidence for unpruned " + "tree!");
            } else {
                m_CF = (new Float(confidenceString)).floatValue();
                if ((m_CF <= 0) || (m_CF >= 1)) {
                    throw new Exception(
                            "Confidence has to be greater than zero and smaller " + "than one!");
                }
            }
        } else {
            m_CF = 0.25f;
        }
        String numFoldsString = Utils.getOption('N', options);
        if (numFoldsString.length() != 0) {
            if (!m_reducedErrorPruning) {
                throw new Exception("Setting the number of folds"
                        + " doesn't make sense if"
                        + " reduced error pruning is not selected.");
            } else {
                m_numFolds = Integer.parseInt(numFoldsString);
            }
        } else {
            m_numFolds = 3;
        }
        String seedString = Utils.getOption('Q', options);
        if (seedString.length() != 0) {
            m_Seed = Integer.parseInt(seedString);
        } else {
            m_Seed = 1;
        }

        super.setOptions(options);

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String[] getOptions() {

        Vector<String> options = new Vector<String>();

        if (m_noCleanup) {
            options.add("-L");
        }
        if (!m_collapseTree) {
            options.add("-O");
        }
        if (m_unpruned) {
            options.add("-U");
        } else {
            if (!m_subtreeRaising) {
                options.add("-S");
            }
            if (m_reducedErrorPruning) {
                options.add("-R");
                options.add("-N");
                options.add("" + m_numFolds);
                options.add("-Q");
                options.add("" + m_Seed);
            } else {
                options.add("-C");
                options.add("" + m_CF);
            }
        }
        if (m_binarySplits) {
            options.add("-B");
        }
        options.add("-M");
        options.add("" + m_minNumObj);

        // NEW PARAMETER **************************************
        options.add("-I");
        options.add("" + m_forestSize);

        options.add("-STG");
        options.add("" + m_strategyNumericAtts);

        options.add("-W");
        options.add("" + m_strategyWeight);

//        options.add("-ERA");
//        options.add("" + m_errorRateAccepted);
        // ****************************************************
        if (m_useLaplace) {
            options.add("-A");
        }
        if (!m_useMDLcorrection) {
            options.add("-J");
        }
        if (m_doNotMakeSplitPointActualValue) {
            options.add("-doNotMakeSplitPointActualValue");
        }

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }

    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier
     */
    @Override
    public String toString() {

        String result = "";

        if (m_root == null) {
            result += "No classifier built";
            return result;
        }

        result += mitProcessing();

        if (m_unpruned) {
            //result += "\nMIT unpruned tree\n------------------\n" + m_root.toString();
            result += "\nMIT NORMALIZED unpruned tree\n------------------\n" + m_root.toStringNormalized(m_forestSize);
            try {
                result += "\nMIT NORMALIZED GRAPH unpruned tree\n------------------\n" + m_root.graphNormalized(m_forestSize);
            } catch (Exception ex) {
                Logger.getLogger(MIT.class.getName()).log(Level.SEVERE, null, ex);
            }
            return result;
        } else {
            //result += "\nMIT pruned tree\n------------------\n" + m_root.toString();
            result += "\nMIT NORMALIZED unpruned tree\n------------------\n" + m_root.toStringNormalized(m_forestSize);

            try {
                result += "\nMIT NORMALIZED GRAPH unpruned tree\n------------------\n" + m_root.graphNormalized(m_forestSize);
            } catch (Exception ex) {
                Logger.getLogger(MIT.class.getName()).log(Level.SEVERE, null, ex);
            }
            return result;
        }

    }

    public String toStringOriginal() {

        if (m_root == null) {
            return "No classifier built";
        }
        if (m_unpruned) {
            return "J48 unpruned tree\n------------------\n" + m_root.toString();
        } else {
            return "J48 pruned tree\n------------------\n" + m_root.toString();
        }
    }

    private String mitProcessing() {

        String result = "";

        if (m_metaInstances != null) {

            result += "\n---------------------------------------------------------------------------\n";

            result += "\n=== META INDUCTION TREE - PROCESSSING ===\n\n";

            result += "\nRandom Forest Size: " + m_forestSize + "\n";

            result += "\nSum of Weights: " + Utils.roundDouble(m_metaInstances.sumOfWeights(), 2) + "\n";

            result += "\n---------------------------------------------------------------------------\n";
        }

        return result;
    }

    /**
     * **************************************************************************************************************
     * NEW CLASSES FOR MIT ALGORITHM
     * **************************************************************************************************************
     */
    enum EnumOperator {

        GreaterThen,
        LessThan,
        EqualTo
    }

    public enum EnumAttributeType {

        Nominal,
        Numeric
    }

    class TreeAsRootToLeaf implements Serializable {

        private List<RootToLeafItem> _treeAsRootToLeaf;
        private int _forestSize;
        public NumericAttributesHandle numericAttributesHandle;
        private DatasetInfo _databaseInfo;

        TreeAsRootToLeaf(RandomTree.Tree tree, int forestSize, Instances originalTrainingData) throws Exception {
            _treeAsRootToLeaf = new ArrayList<>();
            _forestSize = forestSize;
            numericAttributesHandle = new NumericAttributesHandle(originalTrainingData);
            _databaseInfo = new DatasetInfo(originalTrainingData);
            readTreeAsRootToLeaf(tree);
        }

        List<RootToLeafItem> getPathsOfLeafs() {
            return _treeAsRootToLeaf;
        }

        public Double getTreeError() {
            if (_treeAsRootToLeaf.size() > 0) {

                Double aux = _treeAsRootToLeaf.stream().mapToDouble(x -> x.getError()).sum();

                return aux;
            }

            return 0.0;
        }

        private void readTreeAsRootToLeaf(RandomTree.Tree tree) throws Exception {

            Stack<ClassifierIterationStack> stackIteration = new Stack<>();

            dumpTree(0, tree, stackIteration, _treeAsRootToLeaf);
        }

        private void dumpTree(int depth, RandomTree.Tree classfier, Stack<ClassifierIterationStack> stackIteration, List<RootToLeafItem> rootToLeaf) throws Exception {

            // IsLeaf
            if (classfier.getAttribute() == -1) {

                RootToLeafItem item = new RootToLeafItem();

                item.setSubTree(stackIteration);

                item.setClass(classfier);

                // VERIFICAÇÃO PARA ADICIONAR SOMENTE OS EXEMPLOS QUE PESO SEJA > 0
                if (item.getWeight() > 0) {
                    rootToLeaf.add(item);
                }

                if (stackIteration.size() > 0) {
                    stackIteration.pop();
                }

            } else {

                // Nominal
                if (classfier.getInfo().attribute(classfier.getAttribute()).isNominal()) {

                    for (int i = 0; i < classfier.getSuccessors().length; i++) {

                        ClassifierIterationStack stackItem = new ClassifierIterationStack(classfier.getInfo().attribute(classfier.getAttribute()).name(), classfier, i);

                        stackIteration.push(stackItem);

                        dumpTree(depth + 1, classfier.getSuccessors()[i], stackIteration, rootToLeaf);
                    }

                    if (stackIteration.size() > 0) {
                        stackIteration.pop();
                    }

                } else if (classfier.getInfo().attribute(classfier.getAttribute()).isNumeric()) {

                    ClassifierIterationStack stackItem = new ClassifierIterationStack(classfier.getInfo().attribute(classfier.getAttribute()).name(), classfier, EnumOperator.LessThan);

                    stackIteration.push(stackItem);

                    // <  Less
                    dumpTree(depth + 1, classfier.getSuccessors()[0], stackIteration, rootToLeaf);

                    ClassifierIterationStack stackItem2 = new ClassifierIterationStack(classfier.getInfo().attribute(classfier.getAttribute()).name(), classfier, EnumOperator.GreaterThen);

                    stackIteration.push(stackItem2);

                    // >= GreaterOrEqual
                    dumpTree(depth + 1, classfier.getSuccessors()[1], stackIteration, rootToLeaf);

                    if (stackIteration.size() > 0) {
                        stackIteration.pop();
                    }

                }
            }
        }

        @Override
        public String toString() {
            StringBuilder text = new StringBuilder();

            for (RootToLeafItem item : _treeAsRootToLeaf) {
                text.append("\n");

                List<SubTree> wayOfSubTree = item.getPathOfSubTrees();

                String operator = "";

                for (SubTree subTree : wayOfSubTree) {

                    switch (subTree._operator) {
                        case EqualTo:
                            operator = " = ";
                            break;

                        case GreaterThen:
                            operator = " >= ";
                            break;

                        case LessThan:
                            operator = " < ";
                            break;
                        default:
                            operator = " = ";
                    }

                    text.append(subTree.getAttribute() + operator + subTree.getValue() + " -> ");
                }

                text.append(item.getDefinedClass() + " : w = " + item.getWeight() + "; err = " + item.getError() + "; p = " + item.getPrecision() + "; wp = " + item.getWeightPrecision());
            }

            return text.toString();
        }

        class RootToLeafItem implements Serializable {

            private List<SubTree> _pathOfSubTrees;

            private String _definedClass;

            private String _attribute;

            private Double _weight;

            private Double _correct;

            // METRICS
            private Double _precision;
            private Double _laplace;
            private Double _satisfaction;
            private Double _novelty;
            private static final int _noveltyConstant = 100;

            // APPLICATION OF METRICS
            private Double _weightPrecision;
            private Double _weightLaplace;
            private Double _weightSatisfaction;
            private Double _weightNovelty;

            private ContingencyMatrixRootToLeafItem _contingencyMatrix;

            public Double getWeightPrecision() {
                return _weightPrecision;
            }

            private Double _error;

            public Double getPrecision() {
                return _precision;
            }

            private boolean hasNumericAtt;

            RootToLeafItem() {
                _pathOfSubTrees = new ArrayList<SubTree>();
            }

            public Double getCorrect() {
                return _correct;
            }

            public Double getLaplace() {
                return _laplace;
            }

            public Double getSatisfaction() {
                return _satisfaction;
            }

            public Double getNovelty() {
                return _novelty;
            }

            public Double getWeightLaplace() {
                return _weightLaplace;
            }

            public Double getWeightSatisfaction() {
                return _weightSatisfaction;
            }

            public Double getWeightNovelty() {
                return _weightNovelty;
            }

            public ContingencyMatrixRootToLeafItem getContingencyMatrix() {
                return _contingencyMatrix;
            }

            public int getNoveltyConstant() {
                return _noveltyConstant;
            }

            String getDefinedClass() {
                return this._definedClass;
            }

            String getAttribute() {
                return this._attribute;
            }

            Double getWeight() {
                return this._weight;
            }

            Double getError() {
                return this._error;
            }

            List<SubTree> getPathOfSubTrees() {
                return this._pathOfSubTrees;
            }

            SubTree getTreeByAttribute(String attribute) {
                Optional<SubTree> subTree = _pathOfSubTrees.stream().filter(x -> x.getAttribute().equals(attribute)).findFirst();

                if (subTree.isPresent()) {
                    return subTree.get();
                }
                return null;
            }

            public boolean hasNumericAtt() {
                return hasNumericAtt;
            }

            private void setSubTree(String attribute, EnumOperator operator, String value, EnumAttributeType type) {

                if (type == EnumAttributeType.Numeric) {

                    Optional<SubTree> subTreeTest = this._pathOfSubTrees.stream().filter(x -> x.getAttribute().equalsIgnoreCase(attribute)).findFirst();

                    if (subTreeTest.isPresent()) {

                        SubTree subTreeAdded = subTreeTest.get();

                        SubTreeNumeric currentSubTree = (SubTreeNumeric) subTreeAdded;

                        double splitPoint = Double.parseDouble(value);

                        if (operator == EnumOperator.LessThan) {

                            if (splitPoint < currentSubTree.getMaxInterval()) {
                                currentSubTree.setMaxInterval(splitPoint);
                            }

                        } else if (operator == EnumOperator.GreaterThen) {
                            if (splitPoint > currentSubTree.getMinInterval()) {
                                currentSubTree.setMinInterval(splitPoint);
                            }
                        }

                        SubTree subTree = null;

                        subTree = new SubTreeNumeric(attribute, operator, value, type, currentSubTree.getMinInterval(), currentSubTree.getMaxInterval());

                        this._pathOfSubTrees.add(subTree);

                    } else {

                        NumericAttributesHandle.NumericAtrributesItem detailsAtt = numericAttributesHandle.getAtt(attribute);

                        SubTree subTree = null;

                        if (operator == EnumOperator.LessThan) {

                            subTree = new SubTreeNumeric(attribute, operator, value, type, detailsAtt.min, Double.parseDouble(value));

                        } else if (operator == EnumOperator.GreaterThen) {

                            subTree = new SubTreeNumeric(attribute, operator, value, type, Double.parseDouble(value), detailsAtt.max);
                        }

                        this._pathOfSubTrees.add(subTree);
                    }

                } else {

                    SubTree subTree = new SubTreeNominal(attribute, operator, value, type);

                    this._pathOfSubTrees.add(subTree);
                }

            }

            private void setSubTree(Stack<ClassifierIterationStack> stack) {
                for (int i = 0; i < stack.size(); i++) {
                    setSubTree(stack.get(i).getClassifier(), stack.get(i).getIndex(), stack.get(i).getOperator());
                }
            }

            private void setSubTree(RandomTree.Tree classifier, int index, EnumOperator operator) {

                String name = classifier.getInfo().attribute(classifier.getAttribute()).name();

                if (classifier.getInfo().attribute(classifier.getAttribute()).isNominal()) {

                    String value = classifier.getInfo().attribute(classifier.getAttribute()).value(index);

                    setSubTree(name, EnumOperator.EqualTo, value, EnumAttributeType.Nominal);
                } else if (classifier.getInfo().attribute(classifier.getAttribute()).isNumeric()) {

                    hasNumericAtt = true;

                    // For numeric attributes                
                    double splitPoint = classifier.getSplitPoint();
                    setSubTree(name, operator, Double.toString(splitPoint), EnumAttributeType.Numeric);

                }
            }

            void setClass(RandomTree.Tree classifier) {

                double sum = 0, maxCount = 0;
                int maxIndex = 0;
                double classMean = 0;
                double avgError = 0;
                if (classifier.getClassDistribution() != null) { // CASO ENTRAR AQUI, TEM QUE FAZER ALGUNS CALCULOS

                    // NOMINAL
                    if (classifier.getInfo().classAttribute().isNominal()) {
                        sum = Utils.sum(classifier.getClassDistribution());
                        maxIndex = Utils.maxIndex(classifier.getClassDistribution());
                        maxCount = classifier.getClassDistribution()[maxIndex];

                        String definedClass = classifier.getInfo().classAttribute().value(maxIndex);

                        String attribute = classifier.getInfo().classAttribute().name();

                        Double weight = Utils.roundDouble(sum, 2);

                        Double error = Utils.roundDouble(sum - maxCount, 2);

                        setClass(definedClass, weight, error, attribute);

                    } else {

                        // NUMERIC
                        classMean = classifier.getClassDistribution()[0];
                        if (classifier.getDistribution()[1] > 0) {
                            avgError = classifier.getDistribution()[0] / classifier.getDistribution()[1];
                        }

                        String definedClass = Utils.doubleToString(classMean, 2);

                        String attribute = classifier.getInfo().classAttribute().name();

                        Double weight = Utils.roundDouble(classifier.getDistribution()[1], 2);

                        Double error = Utils.roundDouble(avgError, 2);

                        setClass(definedClass, weight, error, attribute);
                    }
                } else {

                    // NOMINAL
                    if (classifier.getInfo().classAttribute().isNominal()) {

                        String definedClass = classifier.getInfo().classAttribute().value(maxIndex);

                        String attribute = classifier.getInfo().classAttribute().name();

                        Double weight = Utils.roundDouble(sum, 2);

                        Double error = Utils.roundDouble(sum - maxCount, 2);

                        setClass(definedClass, weight, error, attribute);

                    } else {

                        // NUMERIC
                        String definedClass = Utils.doubleToString(classMean, 2);

                        String attribute = classifier.getInfo().classAttribute().name();

                        Double weight = Utils.roundDouble(classifier.getDistribution()[1], 2);

                        Double error = Utils.roundDouble(avgError, 2);

                        setClass(definedClass, weight, error, attribute);
                    }
                }
            }

            private void setClass(String definedClass, Double weight, Double error, String attribute) {
                this._definedClass = definedClass;

                // Do normalization                       
                //this._weight = weight > 0 ? Utils.roundDouble(weight / _forestSize, 18) : weight;
                this._weight = weight;
                this._error = error;
                this._correct = (weight - error);
                this._attribute = attribute;
                this._contingencyMatrix = new ContingencyMatrixRootToLeafItem(_databaseInfo);

                int numOfClass = _databaseInfo.getNumClasses();
                double sumOfWeights = _databaseInfo.getSumOfWeights();

                // CALC OF METRICS                                
                ContingencyMatrixRootToLeafItem cm = _contingencyMatrix;
                this._precision = Utils.roundDouble((cm.LR() / cm.l()), 3);
                this._laplace = Utils.roundDouble(((cm.LR() + 1) / (cm.l() + numOfClass)), 3);
                this._novelty = Utils.roundDouble((cm.LR() / sumOfWeights) - ((cm.l() * cm.r()) / (sumOfWeights * sumOfWeights)), 3);
                this._satisfaction = Utils.roundDouble((1 - ((sumOfWeights * cm.L_R()) / (cm.l() * cm._r()))), 3);

                // APPLICATION OF METRICS
                this._weightPrecision = Utils.roundDouble(this._weight * this._precision, 2);
                this._weightLaplace = Utils.roundDouble(this._weight * this._laplace, 2);
                this._weightNovelty = Utils.roundDouble(this._weight * (this._novelty * _noveltyConstant), 2);
                this._weightSatisfaction = Utils.roundDouble(this._weight * this._satisfaction, 2);
            }

            public class ContingencyMatrixRootToLeafItem {

                private DatasetInfo _databaseInfo;

                public ContingencyMatrixRootToLeafItem(DatasetInfo _databaseInfo) {
                    this._databaseInfo = _databaseInfo;
                    _calc();
                }

                private void _calc() {

                    double weightOfClass = this._databaseInfo.getWeightByClassValue(_definedClass).getWeigth();
                    double sumOfWeigths = _databaseInfo.getSumOfWeights();

                    LR = Utils.roundDouble(_correct, 2);
                    L_R = Utils.roundDouble(_error, 2);
                    _LR = Utils.roundDouble((weightOfClass - _correct), 2);
                    _L_R = Utils.roundDouble(((sumOfWeigths - weightOfClass) - _error), 2);

                    l = Utils.roundDouble((LR + L_R), 2);
                    _l = Utils.roundDouble((_LR + _L_R), 2);
                    r = Utils.roundDouble((LR + _LR), 2);
                    _r = Utils.roundDouble((L_R + _L_R), 2);
                }

                private Double LR;
                private Double L_R;
                private Double _LR;
                private Double _L_R;
                private Double l;
                private Double _l;
                private Double r;
                private Double _r;

                public Double LR() {
                    return LR;
                }

                public Double L_R() {
                    return L_R;
                }

                public Double _LR() {
                    return _LR;
                }

                public Double _L_R() {
                    return _L_R;
                }

                public Double l() {
                    return l;
                }

                public Double _l() {
                    return _l;
                }

                public Double r() {
                    return r;
                }

                public Double _r() {
                    return _r;
                }

            }
        }

        abstract class SubTree implements Serializable {

            protected String _attribute;

            protected EnumOperator _operator;

            protected EnumAttributeType _type;

            protected String _value;

            String getAttribute() {
                return _attribute;
            }

            EnumOperator getOperator() {
                return _operator;
            }

            EnumAttributeType getTypeAttribute() {
                return _type;
            }

            String getValue() {
                return _value;
            }

        }

        class SubTreeNominal extends SubTree {

            SubTreeNominal(String attribute, EnumOperator operator, String value, EnumAttributeType type) {
                this._attribute = attribute;
                this._operator = operator;
                this._value = value;
                this._type = type;
            }

        }

        class SubTreeNumeric extends SubTree {

            private double _minInterval;

            public double getMinInterval() {
                return _minInterval;
            }

            public void setMinInterval(double _minInterval) {
                this._minInterval = _minInterval;
            }

            private double _maxInterval;

            public double getMaxInterval() {
                return _maxInterval;
            }

            public double getAVGInterval() {
                return Utils.roundDouble((_minInterval + _maxInterval) / 2, 2);
            }

            public void setMaxInterval(double _maxInterval) {
                this._maxInterval = _maxInterval;
            }

            SubTreeNumeric(String attribute, EnumOperator operator, String value, EnumAttributeType type, double minInterval, double maxInterval) {
                this._attribute = attribute;
                this._operator = operator;
                this._value = value;
                this._type = type;
                this._minInterval = minInterval;
                this._maxInterval = maxInterval;
            }

        }

        class ClassifierIterationStack implements Serializable {

            RandomTree.Tree _classifier;

            int _index;

            EnumOperator _operator;

            String _name;

            ClassifierIterationStack(String name, RandomTree.Tree classifier, int index) {
                this._name = name;
                this._classifier = classifier;
                this._index = index;
            }

            ClassifierIterationStack(String name, RandomTree.Tree classifier, EnumOperator operator) {
                this._name = name;
                this._classifier = classifier;
                this._index = 0;
                this._operator = operator;
            }

            RandomTree.Tree getClassifier() {
                return _classifier;
            }

            int getIndex() {
                return _index;
            }

            EnumOperator getOperator() {
                return _operator;
            }
        }

        class NumericAttributesHandle {

            private List<NumericAtrributesItem> _source;

            NumericAttributesHandle(Instances instances) {
                _source = new ArrayList<NumericAtrributesItem>();
                readAtts(instances);

            }

            private void readAtts(Instances instances) {

                List<Attribute> atts = Collections.list(instances.enumerateAttributes());

                List<Attribute> numericAtts;

                numericAtts = atts.stream().filter(x -> x.isNumeric()).collect(Collectors.toList());

                for (Attribute att : numericAtts) {
                    AttributeStats stats = instances.attributeStats(att.index());

                    NumericAtrributesItem item = new NumericAtrributesItem();

                    item.name = att.name();
                    item.max = stats.numericStats.max;
                    item.min = stats.numericStats.min;
                    item.mean = Utils.roundDouble((item.max + item.min) / 2, 2);
                    _source.add(item);
                }

            }

            NumericAtrributesItem getAtt(String name) {

                Optional<NumericAtrributesItem> att = _source.stream().filter(x -> x.name.equals(name)).findFirst();

                if (att.isPresent()) {
                    return att.get();
                }

                return null;
            }

            List<NumericAtrributesItem> getAll() {
                return _source;
            }

            class NumericAtrributesItem {

                String name;
                double min;
                double max;
                double mean;
            }
        }

    }

    class TreeAsARFF implements Serializable {

        private Instances _instancesMeta;
        private String _strategyNumericAtts;
        private String _strategyWeight;

        public TreeAsARFF(TreeAsRootToLeaf treeAsRootToLeaf, Instances originalTrainingData, String strategyNumericAtts, String strategyWeight) throws Exception {
            _strategyNumericAtts = strategyNumericAtts;
            _strategyWeight = strategyWeight;
            readTreeAsARFF(treeAsRootToLeaf, originalTrainingData);
        }

        public Instances getInstancesMeta() {
            return _instancesMeta;
        }

        private void readTreeAsARFF(TreeAsRootToLeaf treeAsRootToLeaf, Instances originalTrainingData) throws Exception {

            // COPY ORIGINAL DATA TRAINING
            _instancesMeta = new Instances(originalTrainingData);

            _instancesMeta.clear();

            List<Attribute> attributes = Collections.list(_instancesMeta.enumerateAttributes());

            // DEFINE STRATEGY: AVG OR INTERVAL
            boolean isIntervalStrategy = _strategyNumericAtts.equals("I") ? true : false;

            // LINES
            for (TreeAsRootToLeaf.RootToLeafItem rootToLeafItem : treeAsRootToLeaf.getPathsOfLeafs()) {

                //TREATMENT FOR NOVELTY AND SATISFACTION METRIC
                if (_strategyWeight.equals("N") && rootToLeafItem.getNovelty() <= 0) {
                    continue;
                }

                if (_strategyWeight.equals("S") && rootToLeafItem.getSatisfaction() <= 0) {
                    continue;
                }
                //------------------------------------------------------------------------------

                if (rootToLeafItem.getWeight() > 0) {

                    // IF THERE IS A NUMERIC ATTRIBUTE, THE TREATMENT IS DIFFERENT
                    if (rootToLeafItem.hasNumericAtt()) {

                        // FOR THE INTERVAL STRATEGY
                        if (isIntervalStrategy) {

                            // FOR THE INTERVAL STRATEGY
                            _instancesMeta.add(new DenseInstance(_instancesMeta.numAttributes()));

                            Instance newInstanceBeginingInterval = _instancesMeta.instance(_instancesMeta.size() - 1);

                            _instancesMeta.add(new DenseInstance(_instancesMeta.numAttributes()));

                            Instance newInstanceEndInterval = _instancesMeta.instance(_instancesMeta.size() - 1);

                            // COLUMNS
                            for (Attribute att : attributes) {

                                TreeAsRootToLeaf.SubTree subTree = rootToLeafItem.getTreeByAttribute(att.name());

                                if (subTree != null) {

                                    // IF IS NOMINAL, IS THE SAME TREATMENT                            
                                    if (subTree.getTypeAttribute() == EnumAttributeType.Nominal) {
                                        TreeAsRootToLeaf.SubTreeNominal sbNominal = (TreeAsRootToLeaf.SubTreeNominal) subTree;

                                        newInstanceBeginingInterval.setValue(att, sbNominal.getValue());
                                        newInstanceEndInterval.setValue(att, sbNominal.getValue());

                                    } else {

                                        // IF IS NUMERIC AND STRATEGY IS AVG, HAVE TO SET AVG OF INTERVAL
                                        TreeAsRootToLeaf.SubTreeNumeric sbNumeric = (TreeAsRootToLeaf.SubTreeNumeric) subTree;

                                        newInstanceBeginingInterval.setValue(att, sbNumeric.getMinInterval());
                                        newInstanceEndInterval.setValue(att, sbNumeric.getMaxInterval());

                                    }

                                } else {
                                    newInstanceBeginingInterval.setMissing(att);
                                    newInstanceEndInterval.setMissing(att);
                                }
                            }

                            newInstanceBeginingInterval.setClassValue(rootToLeafItem.getDefinedClass());
                            newInstanceEndInterval.setClassValue(rootToLeafItem.getDefinedClass());

                            // ORIGINAL
                            //newInstanceBeginingInterval.setWeight(Utils.roundDouble(rootToLeafItem.getWeight() / 2, 2));
                            //newInstanceEndInterval.setWeight(Utils.roundDouble(rootToLeafItem.getWeight() / 2, 2));
                            newInstanceBeginingInterval.setWeight(Utils.roundDouble(GetWeightByParam(rootToLeafItem) / 2, 2));
                            newInstanceEndInterval.setWeight(Utils.roundDouble(GetWeightByParam(rootToLeafItem) / 2, 2));

                        } else {

                            // FOR THE AVG STRATEGY
                            _instancesMeta.add(new DenseInstance(_instancesMeta.numAttributes()));

                            Instance newInstance = _instancesMeta.instance(_instancesMeta.size() - 1);

                            // COLUMNS
                            for (Attribute att : attributes) {

                                TreeAsRootToLeaf.SubTree subTree = rootToLeafItem.getTreeByAttribute(att.name());

                                if (subTree != null) {

                                    // IF IS NOMINAL, IS THE SAME TREATMENT                            
                                    if (subTree.getTypeAttribute() == EnumAttributeType.Nominal) {
                                        TreeAsRootToLeaf.SubTreeNominal sbNominal = (TreeAsRootToLeaf.SubTreeNominal) subTree;

                                        newInstance.setValue(att, sbNominal.getValue());

                                    } else {

                                        // IF IS NUMERIC AND STRATEGY IS AVG, HAVE TO SET AVG OF INTERVAL
                                        TreeAsRootToLeaf.SubTreeNumeric sbNumeric = (TreeAsRootToLeaf.SubTreeNumeric) subTree;

                                        newInstance.setValue(att, sbNumeric.getAVGInterval());
                                    }

                                } else {
                                    newInstance.setMissing(att);
                                }
                            }
                            // reavaliar
                            newInstance.setClassValue(rootToLeafItem.getDefinedClass());

                            //ORIGINAL                            
                            //newInstance.setWeight(rootToLeafItem.getWeight());
                            newInstance.setWeight(GetWeightByParam(rootToLeafItem));
                        }

                    } else {

                        // FOR THE NOMINAL ATTS, IT IS THE FIRST AND SIMPLE WAY
                        _instancesMeta.add(new DenseInstance(_instancesMeta.numAttributes()));

                        Instance newInstance = _instancesMeta.instance(_instancesMeta.size() - 1);

                        // COLUMNS
                        for (Attribute att : attributes) {

                            TreeAsRootToLeaf.SubTree subTree = rootToLeafItem.getTreeByAttribute(att.name());

                            if (subTree != null) {

                                newInstance.setValue(att, subTree.getValue());

                            } else {
                                newInstance.setMissing(att);
                            }
                        }

                        newInstance.setClassValue(rootToLeafItem.getDefinedClass());

                        // ORIGINAL
                        //newInstance.setWeight(rootToLeafItem.getWeight());
                        newInstance.setWeight(GetWeightByParam(rootToLeafItem));
                    }

                }
            }
        }

        private Double GetWeightByParam(TreeAsRootToLeaf.RootToLeafItem rootToLeafItem) {

            // DEFINE STRATEGY: P = PRECISION, L = LAPLACE, N = NOVELTY, S = SATISFACTION
            if (_strategyWeight.equals("P")) {

                return rootToLeafItem.getWeightPrecision();

            } else if (_strategyWeight.equals("L")) {

                return rootToLeafItem.getWeightLaplace();

            } else if (_strategyWeight.equals("N")) {

                return rootToLeafItem.getWeightNovelty();

            } else if (_strategyWeight.equals("S")) {

                return rootToLeafItem.getWeightSatisfaction();

            } else {

                // PRECISION IS DEFAULT
                return rootToLeafItem.getWeightPrecision();
            }
        }

        @Override
        public String toString() {

            return _instancesMeta != null ? _instancesMeta.toString() : "";
        }

    }

    class TreeItem implements Serializable {

        private RandomTree.Tree _tree;
        private TreeAsRootToLeaf _treeAsRootToLeafNorm;
        private TreeAsARFF _treeAsARFF;

        TreeItem(RandomTree.Tree tree, TreeAsRootToLeaf treeAsRootToLeafNorm, TreeAsARFF treeAsARFF) {
            this._tree = tree;
            this._treeAsRootToLeafNorm = treeAsRootToLeafNorm;
            this._treeAsARFF = treeAsARFF;
        }

        RandomTree.Tree getTree() {
            return this._tree;
        }

        TreeAsRootToLeaf getTreeAsRootToLeaf() {
            return this._treeAsRootToLeafNorm;
        }

        public TreeAsARFF getTreeAsARFF() {
            return _treeAsARFF;
        }
    }

    class TreeAsTable implements Serializable {

        private int sizeOfColumn = 15;

        private List<TreeAsTableRow> _rows;

        TreeAsTable(TreeAsRootToLeaf treeAsRootToLeaf, Instances originalTrainingData) throws Exception {

            _rows = new ArrayList<>();
            readTreeAsTable(treeAsRootToLeaf, originalTrainingData);
        }

        List<TreeAsTableRow> getRows() {
            return _rows;
        }

        private void readTreeAsTable(TreeAsRootToLeaf treeAsRootToLeaf, Instances originalTrainingData) throws Exception {

            List<Attribute> attributes = Collections.list(originalTrainingData.enumerateAttributes());
            // lines
            for (TreeAsRootToLeaf.RootToLeafItem rootToLeafItem : treeAsRootToLeaf.getPathsOfLeafs()) {

                TreeAsTableRow row = new TreeAsTableRow();

                for (Attribute att : attributes) {

                    Boolean isClass = originalTrainingData.classIndex() == att.index();

                    if (!isClass) {

                        TreeAsRootToLeaf.SubTree subTree = rootToLeafItem.getTreeByAttribute(att.name());

                        if (subTree != null) {

                            if (subTree.getTypeAttribute() == EnumAttributeType.Nominal) {

                                TreeAsRootToLeaf.SubTreeNominal sbNominal = (TreeAsRootToLeaf.SubTreeNominal) subTree;

                                if (sbNominal != null) {
                                    row.addColumnNominal(sbNominal.getAttribute(), sbNominal.getOperator(), sbNominal.getValue());
                                }

                            } else {

                                TreeAsRootToLeaf.SubTreeNumeric sbNumeric = (TreeAsRootToLeaf.SubTreeNumeric) subTree;

                                if (sbNumeric != null) {
                                    row.addColumnNumeric(sbNumeric.getAttribute(), sbNumeric.getOperator(), sbNumeric.getValue(), sbNumeric.getMinInterval(), sbNumeric.getMaxInterval());
                                }
                            }

                        } else {
                            row.addColumnNominal(att.name(), EnumOperator.EqualTo, "?");
                        }

                    }
                }

                row.setAttribute(rootToLeafItem.getAttribute());
                row.setDefinedClass(rootToLeafItem.getDefinedClass());
                row.setError(rootToLeafItem.getError());
                row.setWeight(rootToLeafItem.getWeight());

                row.setCorrect(rootToLeafItem.getCorrect());
                row.setPrecision(rootToLeafItem.getPrecision());
                row.setLaplace(rootToLeafItem.getLaplace());
                row.setNovelty(rootToLeafItem.getNovelty());
                row.setNoveltyConstant(rootToLeafItem.getNoveltyConstant());
                row.setSatisfaction(rootToLeafItem.getSatisfaction());

                row.setWeightPrecision(rootToLeafItem.getWeightPrecision());
                row.setWeightLaplace(rootToLeafItem.getWeightLaplace());
                row.setWeightNovelty(rootToLeafItem.getWeightNovelty());
                row.setWeightSatisfaction(rootToLeafItem.getSatisfaction());

                row.setContingencyMatrix(rootToLeafItem.getContingencyMatrix());

                _rows.add(row);
            }

        }

        @Override
        public String toString() {
            StringBuilder text = new StringBuilder();

            text.append("\n");

            // HEADER
            if (_rows.size() > 0) {

                TreeAsTableRow firstRow = _rows.get(0);

                List<TreeAsTableRow.TreeAsTableColumn> cols = firstRow.getColumns();

                for (TreeAsTableRow.TreeAsTableColumn col : cols) {
                    text.append(padRight(col.getAttribute(), sizeOfColumn));
                }

                text.append(padRight(firstRow.getClassAttribute(), sizeOfColumn));

                text.append(padRight("weight", 10));

                text.append(padRight("error", 10));

                text.append(padRight("correct", 10));

                text.append(padRight("LR", 10));

                text.append(padRight("L_R", 10));

                text.append(padRight("_LR", 10));

                text.append(padRight("_L_R", 10));

                text.append(padRight("l", 10));

                text.append(padRight("_l", 10));

                text.append(padRight("r", 10));

                text.append(padRight("_r", 10));

                text.append(padRight("novelty (*" + firstRow.getNoveltyConstant() + ")", 18));

                text.append(padRight("satisfaction", sizeOfColumn));

                text.append(padRight("precision", sizeOfColumn));

                text.append(padRight("laplace", sizeOfColumn));

                text.append(padRight("w_novelty", sizeOfColumn));

                text.append(padRight("w_satisfaction", sizeOfColumn));

                text.append(padRight("w_precision", sizeOfColumn));

                text.append(padRight("w_laplace", sizeOfColumn));

                text.append("\n");

            } else {
                return "";
            }

            //BODY
            for (TreeAsTableRow row : _rows) {

                List<TreeAsTableRow.TreeAsTableColumn> cols = row.getColumns();

                for (TreeAsTableRow.TreeAsTableColumn col : cols) {

                    if (col.isNumeric()) {

                        TreeAsTableRow.TreeAsTableColumnNumeric colNumeric = (TreeAsTableRow.TreeAsTableColumnNumeric) col;

                        String intervalText = "[" + colNumeric.getMinInterval() + "," + colNumeric.getMaxInterval() + "]";

                        text.append(padRight(intervalText, sizeOfColumn));
                    } else {
                        text.append(padRight(col.getValue(), sizeOfColumn));
                    }
                }

                text.append(padRight(row.getDefinedClass(), sizeOfColumn));

                text.append(padRight(row.getWeight().toString(), 10));

                text.append(padRight(row.getError().toString(), 10));

                text.append(padRight(row.getCorrect().toString(), 10));

                text.append(padRight(row.getContingencyMatrix().LR().toString(), 10));

                text.append(padRight(row.getContingencyMatrix().L_R().toString(), 10));

                text.append(padRight(row.getContingencyMatrix()._LR().toString(), 10));

                text.append(padRight(row.getContingencyMatrix()._L_R().toString(), 10));

                text.append(padRight(row.getContingencyMatrix().l().toString(), 10));

                text.append(padRight(row.getContingencyMatrix()._l().toString(), 10));

                text.append(padRight(row.getContingencyMatrix().r().toString(), 10));

                text.append(padRight(row.getContingencyMatrix()._r().toString(), 10));

                Double novelty_x_Constant = Utils.roundDouble((row.getNovelty() * row.getNoveltyConstant()), 2);

                text.append(padRight(row.getNovelty().toString() + " (" + novelty_x_Constant.toString() + ")", 18));

                text.append(padRight(row.getSatisfaction().toString(), sizeOfColumn));

                text.append(padRight(row.getPrecision().toString(), sizeOfColumn));

                text.append(padRight(row.getLaplace().toString(), sizeOfColumn));

                text.append(padRight(row.getWeightNovelty().toString(), sizeOfColumn));

                text.append(padRight(row.getWeightSatisfaction().toString(), sizeOfColumn));

                text.append(padRight(row.getWeightPrecision().toString(), sizeOfColumn));

                text.append(padRight(row.getWeightLaplace().toString(), sizeOfColumn));

                text.append("\n");
            }

            return text.toString();
        }

        String padRight(String s, int n) {
            return String.format("%-" + n + "s", s);
        }

        String padLeft(String s, int n) {
            return String.format("%" + n + "s", s);
        }

        class TreeAsTableRow implements Serializable {

            private List<TreeAsTableColumn> _columns;

            private String _definedClass;

            private String _classAttribute;

            private Double _weight;

            private Double _error;

            // CALCULED VALUES 
            private Double _correct;
            private Double _precision;
            private Double _laplace;
            private Double _satisfaction;
            private Double _novelty;
            private int _noveltyConstant;
            private Double _weightPrecision;
            private Double _weightLaplace;
            private Double _weightSatisfaction;
            private Double _weightNovelty;
            private TreeAsRootToLeaf.RootToLeafItem.ContingencyMatrixRootToLeafItem _contingencyMatrix;

            TreeAsTableRow() {
                this._columns = new ArrayList<>();
            }

            void addColumnNominal(String attribute, EnumOperator operator, String value) {

                TreeAsTableColumn col = new TreeAsTableColumnNominal(attribute, operator, value);

                _columns.add(col);
            }

            void addColumnNumeric(String attribute, EnumOperator operator, String value, double minInterval, double maxInterval) {

                TreeAsTableColumn col = new TreeAsTableColumnNumeric(attribute, operator, value, minInterval, maxInterval);

                _columns.add(col);
            }

            List<TreeAsTableColumn> getColumns() {
                return this._columns;
            }

            void setDefinedClass(String definedClass) {
                this._definedClass = definedClass;
            }

            void setAttribute(String attribute) {
                this._classAttribute = attribute;
            }

            void setWeight(Double weight) {
                this._weight = weight;
            }

            void setError(Double error) {
                this._error = error;
            }

            String getDefinedClass() {
                return _definedClass;
            }

            String getClassAttribute() {
                return _classAttribute;
            }

            Double getWeight() {
                return _weight;
            }

            Double getError() {
                return _error;
            }

            public Double getCorrect() {
                return _correct;
            }

            public void setCorrect(Double _correct) {
                this._correct = _correct;
            }

            public Double getPrecision() {
                return _precision;
            }

            public void setPrecision(Double _precision) {
                this._precision = _precision;
            }

            public Double getLaplace() {
                return _laplace;
            }

            public void setLaplace(Double _laplace) {
                this._laplace = _laplace;
            }

            public Double getSatisfaction() {
                return _satisfaction;
            }

            public void setSatisfaction(Double _satisfaction) {
                this._satisfaction = _satisfaction;
            }

            public Double getNovelty() {
                return _novelty;
            }

            public void setNovelty(Double _novelty) {
                this._novelty = _novelty;
            }

            public int getNoveltyConstant() {
                return _noveltyConstant;
            }

            public void setNoveltyConstant(int _noveltyConstant) {
                this._noveltyConstant = _noveltyConstant;
            }

            public Double getWeightPrecision() {
                return _weightPrecision;
            }

            public void setWeightPrecision(Double _weightPrecision) {
                this._weightPrecision = _weightPrecision;
            }

            public Double getWeightLaplace() {
                return _weightLaplace;
            }

            public void setWeightLaplace(Double _weightLaplace) {
                this._weightLaplace = _weightLaplace;
            }

            public Double getWeightSatisfaction() {
                return _weightSatisfaction;
            }

            public void setWeightSatisfaction(Double _weightSatisfaction) {
                this._weightSatisfaction = _weightSatisfaction;
            }

            public Double getWeightNovelty() {
                return _weightNovelty;
            }

            public void setWeightNovelty(Double _weightNovelty) {
                this._weightNovelty = _weightNovelty;
            }

            public TreeAsRootToLeaf.RootToLeafItem.ContingencyMatrixRootToLeafItem getContingencyMatrix() {
                return _contingencyMatrix;
            }

            public void setContingencyMatrix(TreeAsRootToLeaf.RootToLeafItem.ContingencyMatrixRootToLeafItem _contingencyMatrix) {
                this._contingencyMatrix = _contingencyMatrix;
            }

            abstract class TreeAsTableColumn implements Serializable {

                protected boolean _isNumeric = false;

                protected String _attribute;

                protected EnumOperator _operator;

                protected String _value;

                public String getAttribute() {
                    return _attribute;
                }

                public EnumOperator getOperator() {
                    return _operator;
                }

                public String getValue() {
                    return _value;
                }

                public boolean isNumeric() {
                    return _isNumeric;
                }
            }

            class TreeAsTableColumnNominal extends TreeAsTableColumn implements Serializable {

                TreeAsTableColumnNominal(String attribute, EnumOperator operator, String value) {
                    this._attribute = attribute;
                    this._operator = operator;
                    this._value = value;
                    this._isNumeric = false;
                }
            }

            class TreeAsTableColumnNumeric extends TreeAsTableColumn implements Serializable {

                double _minInterval;

                public double getMinInterval() {
                    return _minInterval;
                }

                public void setMinInterval(double _minInterval) {
                    this._minInterval = _minInterval;
                }

                double _maxInterval;

                public double getMaxInterval() {
                    return _maxInterval;
                }

                public void setMaxInterval(double _maxInterval) {
                    this._maxInterval = _maxInterval;
                }

                TreeAsTableColumnNumeric(String attribute, EnumOperator operator, String value, double minInterval, double maxInterval) {
                    this._attribute = attribute;
                    this._operator = operator;
                    this._value = value;
                    this._minInterval = minInterval;
                    this._maxInterval = maxInterval;
                    this._isNumeric = true;
                }
            }
        }

    }

    class DatasetInfo implements Serializable {

        Instances _ins;
        List<WeightByClassValueItem> _weightByClassValueList;

        public DatasetInfo(Instances _ins) {
            this._ins = _ins;
            this._weightByClassValueList = new ArrayList<>();
            this.calcWeightByClassValue();
        }

        private void calcWeightByClassValue() {

            Attribute attrClass = _ins.classAttribute();

            AttributeStats statsAttrClass = _ins.attributeStats(attrClass.index());

            List<Object> values = Collections.list(attrClass.enumerateValues());

            for (int i = 0; i < values.size(); i++) {

                Object object = values.get(i);

                double weigth = statsAttrClass.nominalWeights[i];

                this._weightByClassValueList.add(new WeightByClassValueItem(object.toString(), weigth));
            }
        }

        public WeightByClassValueItem getWeightByClassValue(String value) {
            Optional<WeightByClassValueItem> aux = this._weightByClassValueList.stream().filter(x -> x.getValueClass().equals(value)).findFirst();

            if (aux.isPresent()) {
                return aux.get();
            }
            return null;
        }

        public int getNumAttributes() {
            return _ins.numAttributes();
        }

        public int getNumClasses() {
            return _ins.numClasses();
        }

        public int getNumInstances() {
            return _ins.numInstances();
        }

        public double getSumOfWeights() {
            return _ins.sumOfWeights();
        }

        public class WeightByClassValueItem {

            private String _valueClass;
            private double _weigth;

            public WeightByClassValueItem(String _valueClass, double _weigth) {
                this._valueClass = _valueClass;
                this._weigth = _weigth;
            }

            public String getValueClass() {
                return _valueClass;
            }

            public double getWeigth() {
                return _weigth;
            }
        }
    }
}
