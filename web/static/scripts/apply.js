////$(document).ready(function(){
////    $("#submitVis").click(function() {
////         setTimeout(function(){
////                $("#image").append( '<img src="/static/vis/01804.jpg" class="img-responsive well" alt="Cinque Terre" width="400" height="380">' );
////         }, 1000);
////    });
////    $("#submitVis").click(function() {
////         setTimeout(function(){
////                $("#pred").append('<ul><li>animals: 1.82%</li><li>cars: 96.13%</li><li>people: 2.05%</li></ul>' );
////         }, 4000);
////    });
////});
////
////
////$(document).ready(function(e) {
////    $("#uploadimage").on('submit', (function(e) {
////        //console.log("")
////        //formdata.append("image", "value")
////        e.preventDefault();
////        $("#message").empty();
////        $('#loading').show();
////        $.ajax({
////            url: "/home/", // Url to which the request is send
////            type: "POST", // Type of request to be send, called as method
////            data: new formdata(this), // Data sent to server, a set of key/value pairs (i.e. form fields and values)
////            contentType: false, // The content type used when sending data to the server.
////            cache: false, // To unable request pages to be cached
////            processData: false, // To send DOMDocument or non processed data file it is set to false
////            success: function(data)  // A function to be called if request succeeds
////            {
////                $('#loading').hide();
////                $("#message").html(data);
////            }
////        });
////    }));
////
////    // Function to preview image after validation
////    $(function() {
////        $("#file").change(function() {
////            $("#message").empty(); // To remove the previous error message
////            var file = this.files[0];
////            var imagefile = file.type;
////            var match = ["image/jpeg", "image/png", "image/jpg"];
////            if (!((imagefile == match[0]) || (imagefile == match[1]) || (imagefile == match[2])))
////            {
////                $('#previewing').attr('src', 'noimage.png');
////                $("#message").html("<p id='error'>Please Select A valid Image File</p>" + "<h4>Note</h4>" + "<span id='error_message'>Only jpeg, jpg and png Images type allowed</span>");
////                return false;
////            }
////            else
////            {
////                var reader = new FileReader();
////                reader.onload = imageIsLoaded;
////                reader.readAsDataURL(this.files[0]);
////            }
////        });
////    });
////    function imageIsLoaded(e) {
////        $("#file").css("color", "green");
////        $('#image_preview').css("display", "block");
////        $('#previewing').attr('src', e.target.result);
////        $('#previewing').attr('width', '250px');
////        $('#previewing').attr('height', '230px');
////    }
////    ;
//
//function start() {
//
////    var result = undefined;
////    $("#submit").click(function() {
////        console.log('image is loded');
////        if (!!result) {
////            $.ajax({
////                url: "/homeee/",
////                type: "POST",
////                data: result,
////                success: function(result) {
////                    $("#result").val(result);
////                }
////            });
////        }
////
////    });
//
//    $("#selectFiles").change(function(e) {
//        onChange(e);
//    });
//
//    function onChange(event) {
//     console.log('file is change')
//        var reader = new FileReader();
//        reader.onload = onReaderLoad;
//        reader.readAsText(event.target.files[0]);
//    }
//
//    function onReaderLoad(event){
//        //alert(event.target.result);
//        var result = new FileReader();
//        result.onload = imageIsLoaded;
//        result.readAsDataURL(this.files[0]);
//        console.log('image is loded');
////        $('#json_content').val(event.target.result);
////        result = JSON.parse(event.target.result);
////        console.log('json load');
//    }
//
//    function imageIsLoaded(e) {
//        console.log('image is loaded')
//    }
//
//}
//
//$(document).ready(function(){
//    start();
//});
//
//

$(document).ready(function (e) {
    $('#uploadimage').on('submit',(function(e) {
        e.preventDefault();
        var formData = new FormData($('#uploadimage').get(0));
        //formData.append('name', $('#uploadimage').get(0));
//        var formData = new FormData(this)new;
        console.log(formData)
        $.ajax({
            type:'POST',
            url: '/apply/test/',
            data:formData,
            cache:false,
            contentType: false,
            processData: false,
            success:function(data){
                console.log("success");
                console.log(data);
            },
            error: function(data){
                console.log("error");
                console.log(data);
            }
        });
    }));

    $("#selectFiles").on("change", function() {
        console.log('change')
        $("#imageUploadForm").submit();
    });
});